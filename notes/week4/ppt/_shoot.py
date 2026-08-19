from playwright.sync_api import sync_playwright
import os, urllib.parse

OUT = r"C:\Users\19374\Desktop\AI Infra\notes\week4\ppt\_shots"
os.makedirs(OUT, exist_ok=True)
url_base = "file:///C:/Users/19374/Desktop/AI%20Infra/notes/week4/ppt/index.html"
# 抽查页: 封面/过渡/图文/大表/split/dark推导/four-cards/时间线/S22/三图/接力/展望/收尾
slides = [1, 2, 3, 5, 10, 13, 15, 21, 24, 28, 30, 34, 36, 38, 39]

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1920, "height": 1080})
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    for n in slides:
        page.goto(f"{url_base}?slide={n}")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(2300)  # 等入场动效稳定
        page.screenshot(path=os.path.join(OUT, f"slide-{n:02d}.png"))
        print(f"shot slide {n}")
    browser.close()
    if errors:
        print("CONSOLE ERRORS:")
        for e in errors[:20]:
            print(" -", e)
    else:
        print("no console errors")
