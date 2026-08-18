/* _fastenc: C fast path for the collection hot loop's zone/row encoding.
 *
 * Exports fill_zone(rows, base, objs, n, viewer) -> next_base, the C twin of
 * fishrl.obs.encoder._fill_zone_py. Per occupied slot it does exactly what the
 * Python loop does -- read the CardInstance attributes, build the row-cache key,
 * look it up in the SHARED Python-side _ROW_CACHE, memcpy the cached row into the
 * output -- but without interpreter dispatch (~63 rows/decision made this the
 * hottest loop in collection even after the Python row cache).
 *
 * Contract with fishrl.obs.encoder (enforced by setup() + test_fastenc_equivalence):
 *   - the cache dict and the miss-path builder are the PYTHON objects, passed in
 *     via setup(); this module never inserts into the cache (_build_row is the
 *     only writer), so C and Python paths cannot diverge on cache content;
 *   - key tuples are built element-for-element like the Python expression
 *       (name, type_line, oracle_text, text_variant(o) if o.text_changes else None,
 *        power, toughness, damage_marked, sum(counters.values()) if counters else 0,
 *        bool(tapped), bool(entered_this_turn), controller == viewer)
 *     with text_variant / the counters sum delegated to the passed-in Python
 *     callables (both rare paths), so hashes/equality match Python-built keys;
 *   - `rows` must be a C-contiguous float32 buffer of CARD_F-wide rows, zeroed in
 *     the slots this call fills; a None obj encodes the hidden slot (row[N_NAMES]=1).
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>

static PyObject *g_cache = NULL;        /* fishrl.obs.encoder._ROW_CACHE (dict)   */
static PyObject *g_build_row = NULL;    /* fishrl.obs.encoder._build_row          */
static PyObject *g_text_variant = NULL; /* forgetful_fish.state.text_variant      */
static PyObject *g_sum_values = NULL;   /* lambda-equivalent: sum(d.values())     */
static Py_ssize_t g_card_f = 0;
static Py_ssize_t g_n_names = 0;

/* interned attribute names */
static PyObject *s_name, *s_type_line, *s_oracle_text, *s_text_changes,
    *s_power, *s_toughness, *s_damage_marked, *s_counters,
    *s_tapped, *s_entered_this_turn, *s_controller;

static PyObject *
fastenc_setup(PyObject *self, PyObject *args)
{
    PyObject *cache, *build_row, *text_variant, *sum_values;
    Py_ssize_t card_f, n_names;
    if (!PyArg_ParseTuple(args, "O!OOOnn", &PyDict_Type, &cache, &build_row,
                          &text_variant, &sum_values, &card_f, &n_names))
        return NULL;
    Py_XSETREF(g_cache, Py_NewRef(cache));
    Py_XSETREF(g_build_row, Py_NewRef(build_row));
    Py_XSETREF(g_text_variant, Py_NewRef(text_variant));
    Py_XSETREF(g_sum_values, Py_NewRef(sum_values));
    g_card_f = card_f;
    g_n_names = n_names;
    Py_RETURN_NONE;
}

/* Build the 11-element key tuple for `o` (new ref), or NULL on error. */
static PyObject *
row_key(PyObject *o, PyObject *viewer)
{
    PyObject *name = NULL, *tl = NULL, *ot = NULL, *chg = NULL, *tv = NULL;
    PyObject *power = NULL, *tough = NULL, *dmg = NULL, *cnt = NULL, *csum = NULL;
    PyObject *tapped = NULL, *entered = NULL, *ctrl = NULL, *ctrl_eq = NULL;
    PyObject *key = NULL;
    int truth;

    if (!(name = PyObject_GetAttr(o, s_name))) goto done;
    if (!(tl = PyObject_GetAttr(o, s_type_line))) goto done;
    if (!(ot = PyObject_GetAttr(o, s_oracle_text))) goto done;

    /* tv = text_variant(o) if o.text_changes else None  (text changes are rare) */
    if (!(chg = PyObject_GetAttr(o, s_text_changes))) goto done;
    truth = PyObject_IsTrue(chg);
    if (truth < 0) goto done;
    if (truth) {
        if (!(tv = PyObject_CallOneArg(g_text_variant, o))) goto done;
    } else {
        tv = Py_NewRef(Py_None);
    }

    if (!(power = PyObject_GetAttr(o, s_power))) goto done;
    if (!(tough = PyObject_GetAttr(o, s_toughness))) goto done;
    if (!(dmg = PyObject_GetAttr(o, s_damage_marked))) goto done;

    /* csum = sum(counters.values()) if counters else 0  (counters are rare) */
    if (!(cnt = PyObject_GetAttr(o, s_counters))) goto done;
    truth = PyObject_IsTrue(cnt);
    if (truth < 0) goto done;
    if (truth) {
        if (!(csum = PyObject_CallOneArg(g_sum_values, cnt))) goto done;
    } else {
        csum = PyLong_FromLong(0);
        if (!csum) goto done;
    }

    if (!(tapped = PyObject_GetAttr(o, s_tapped))) goto done;
    truth = PyObject_IsTrue(tapped);
    if (truth < 0) goto done;
    Py_SETREF(tapped, Py_NewRef(truth ? Py_True : Py_False));

    if (!(entered = PyObject_GetAttr(o, s_entered_this_turn))) goto done;
    truth = PyObject_IsTrue(entered);
    if (truth < 0) goto done;
    Py_SETREF(entered, Py_NewRef(truth ? Py_True : Py_False));

    if (!(ctrl = PyObject_GetAttr(o, s_controller))) goto done;
    if (!(ctrl_eq = PyObject_RichCompare(ctrl, viewer, Py_EQ))) goto done;

    key = PyTuple_Pack(11, name, tl, ot, tv, power, tough, dmg, csum,
                       tapped, entered, ctrl_eq);
done:
    Py_XDECREF(name); Py_XDECREF(tl); Py_XDECREF(ot); Py_XDECREF(chg);
    Py_XDECREF(tv); Py_XDECREF(power); Py_XDECREF(tough); Py_XDECREF(dmg);
    Py_XDECREF(cnt); Py_XDECREF(csum); Py_XDECREF(tapped); Py_XDECREF(entered);
    Py_XDECREF(ctrl); Py_XDECREF(ctrl_eq);
    return key;
}

/* memcpy a cached row (1-D contiguous float32 ndarray) into dst. */
static int
copy_row(PyObject *row, float *dst)
{
    Py_buffer rb;
    if (PyObject_GetBuffer(row, &rb, PyBUF_SIMPLE) < 0)
        return -1;
    if (rb.len != g_card_f * (Py_ssize_t)sizeof(float)) {
        PyBuffer_Release(&rb);
        PyErr_SetString(PyExc_ValueError, "cached row has wrong byte length");
        return -1;
    }
    memcpy(dst, rb.buf, (size_t)rb.len);
    PyBuffer_Release(&rb);
    return 0;
}

static PyObject *
fastenc_fill_zone(PyObject *self, PyObject *args)
{
    PyObject *rows, *objs, *viewer;
    Py_ssize_t base, n;
    if (!PyArg_ParseTuple(args, "OnOnU", &rows, &base, &objs, &n, &viewer))
        return NULL;
    if (g_cache == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "_fastenc.setup() has not been called");
        return NULL;
    }

    Py_buffer view;
    if (PyObject_GetBuffer(rows, &view,
                           PyBUF_C_CONTIGUOUS | PyBUF_WRITABLE | PyBUF_FORMAT) < 0)
        return NULL;
    if (view.itemsize != (Py_ssize_t)sizeof(float) ||
        view.format == NULL || view.format[0] != 'f') {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_TypeError, "rows must be a float32 buffer");
        return NULL;
    }
    Py_ssize_t total_rows = view.len / (g_card_f * (Py_ssize_t)sizeof(float));
    if (view.len != total_rows * g_card_f * (Py_ssize_t)sizeof(float) ||
        base < 0 || base + n > total_rows) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_ValueError, "rows buffer / base / n mismatch");
        return NULL;
    }

    PyObject *seq = PySequence_Fast(objs, "objs must be a sequence");
    if (seq == NULL) {
        PyBuffer_Release(&view);
        return NULL;
    }
    Py_ssize_t count = PySequence_Fast_GET_SIZE(seq);
    if (count > n)
        count = n;

    float *buf = (float *)view.buf;
    for (Py_ssize_t i = 0; i < count; i++) {
        PyObject *o = PySequence_Fast_GET_ITEM(seq, i);   /* borrowed */
        float *dst = buf + (base + i) * g_card_f;
        if (o == Py_None) {
            dst[g_n_names] = 1.0f;                        /* hidden/unknown slot */
            continue;
        }
        PyObject *key = row_key(o, viewer);
        if (key == NULL) goto fail;
        PyObject *row = PyDict_GetItemWithError(g_cache, key);   /* borrowed */
        if (row == NULL) {
            if (PyErr_Occurred()) { Py_DECREF(key); goto fail; }
            /* miss: the Python builder constructs, caches (under its cap), returns */
            PyObject *o_key[2] = {o, key};
            row = PyObject_Vectorcall(g_build_row, o_key, 2, NULL);   /* new ref */
            if (row == NULL) { Py_DECREF(key); goto fail; }
            int rc = copy_row(row, dst);
            Py_DECREF(row);
            Py_DECREF(key);
            if (rc < 0) goto fail;
            continue;
        }
        Py_DECREF(key);
        if (copy_row(row, dst) < 0) goto fail;
    }
    Py_DECREF(seq);
    PyBuffer_Release(&view);
    return PyLong_FromSsize_t(base + n);
fail:
    Py_DECREF(seq);
    PyBuffer_Release(&view);
    return NULL;
}

static PyMethodDef fastenc_methods[] = {
    {"setup", fastenc_setup, METH_VARARGS,
     "setup(row_cache, build_row, text_variant, sum_values, card_f, n_names)"},
    {"fill_zone", fastenc_fill_zone, METH_VARARGS,
     "fill_zone(rows, base, objs, n, viewer) -> next base"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef fastenc_module = {
    PyModuleDef_HEAD_INIT, "_fastenc",
    "C fast path for fishrl's zone/row feature encoding.", -1, fastenc_methods,
};

PyMODINIT_FUNC
PyInit__fastenc(void)
{
#define INTERN(var, str) if (!(var = PyUnicode_InternFromString(str))) return NULL
    INTERN(s_name, "name");
    INTERN(s_type_line, "type_line");
    INTERN(s_oracle_text, "oracle_text");
    INTERN(s_text_changes, "text_changes");
    INTERN(s_power, "power");
    INTERN(s_toughness, "toughness");
    INTERN(s_damage_marked, "damage_marked");
    INTERN(s_counters, "counters");
    INTERN(s_tapped, "tapped");
    INTERN(s_entered_this_turn, "entered_this_turn");
    INTERN(s_controller, "controller");
#undef INTERN
    return PyModule_Create(&fastenc_module);
}
