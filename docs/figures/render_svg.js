const { chromium } = require('playwright');
const fs = require('fs');
(async () => {
  const [svgPath, outBase, scale] = process.argv.slice(2);
  const svg = fs.readFileSync(svgPath, 'utf8');
  const W = +svg.match(/width="(\d+)"/)[1], H = +svg.match(/height="(\d+)"/)[1];
  const b = await chromium.launch();
  const pg = await b.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: +(scale||2.5) });
  await pg.setContent(`<html><body style="margin:0">${svg}</body></html>`);
  await pg.screenshot({ path: outBase + '.png' });
  await pg.pdf({ path: outBase + '.pdf', width: W + 'px', height: H + 'px', printBackground: true, pageRanges: '1' });
  await b.close();
})();
