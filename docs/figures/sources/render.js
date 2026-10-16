// Render SVG files to PNG with Chromium. Usage: node render.js a.svg b.svg ...
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');
(async () => {
  const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium', args: ['--no-sandbox'] });
  for (const svgPath of process.argv.slice(2)) {
    const svg = fs.readFileSync(svgPath, 'utf8');
    const w = +svg.match(/<svg[^>]*width="(\d+)"/)[1], h = +svg.match(/<svg[^>]*height="(\d+)"/)[1];
    const page = await browser.newPage({ viewport: { width: w, height: h }, deviceScaleFactor: 1 });
    await page.setContent(`<html><body style="margin:0;background:white">${svg}</body></html>`);
    await page.waitForTimeout(200);
    const out = svgPath.replace(/\.svg$/, '.png');
    await page.screenshot({ path: out, clip: { x: 0, y: 0, width: w, height: h } });
    console.log('wrote', out, w, 'x', h);
    await page.close();
  }
  await browser.close();
})();
