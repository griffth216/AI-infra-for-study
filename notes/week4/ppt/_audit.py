from playwright.sync_api import sync_playwright

base = "file:///C:/Users/19374/Desktop/AI%20Infra/notes/week4/ppt/index.html"

JS = """
(n) => {
  const vh = innerHeight, vw = innerWidth;
  const slide = document.querySelectorAll('.slide')[n-1];
  const issues = [];
  // broken images
  slide.querySelectorAll('img').forEach(img => {
    if (img.naturalWidth === 0) issues.push('BROKEN IMG: ' + img.getAttribute('src'));
  });
  // overflow checks (skip decorative/absolute backgrounds)
  slide.querySelectorAll('*').forEach(el => {
    if (el.matches('canvas, .dot-mat, .ring-mat, .cross-mat')) return;
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden') return;
    const r = el.getBoundingClientRect();
    if (r.width < 4 || r.height < 4) return;
    if (r.bottom > vh * 0.965 && r.top > vh * 0.5) {
      issues.push('BOTTOM OVERFLOW ' + Math.round(r.bottom) + 'px: <' + el.tagName.toLowerCase() +
        ' class="' + (el.className||'').toString().slice(0,40) + '"> ' + (el.textContent||'').trim().slice(0,40));
    }
    if (r.right > vw + 4 || r.left < -4) {
      issues.push('H-OVERFLOW: <' + el.tagName.toLowerCase() + ' class="' + (el.className||'').toString().slice(0,40) + '">');
    }
  });
  return issues.slice(0, 6);
}
"""

with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    pg = b.new_page(viewport={"width": 1920, "height": 1080})
    total_bad = 0
    for n in range(1, 40):
        pg.goto(f"{base}?slide={n}")
        pg.wait_for_load_state("networkidle")
        pg.wait_for_timeout(1600)
        issues = pg.evaluate(JS, n)
        if issues:
            total_bad += 1
            print(f"--- slide {n:02d}:")
            for i in issues:
                print("   ", i)
    b.close()
    print(f"\naudit done: {total_bad} slide(s) with findings" if total_bad else "\naudit done: all 39 slides clean")
