from playwright.async_api import async_playwright
import asyncio
from rich import print

input_sentences = [
    sentences
    for sentences in open(
        "Transliteration\\Qualitative Analysis\\temp_for_google.txt",
        "r",
        encoding="utf-8",
    )
    .read()
    .splitlines()
    if sentences != ""
]

outputs = []

# 163, 201, 200, 211, 219, 222, 231, 232, 239, 240,


async def main():

    async with async_playwright() as p:

        browser = await p.chromium.launch(headless=False)

        page = await browser.new_page()

        await page.goto("https://www.google.com/intl/ne/inputtools/try/")

        await page.wait_for_timeout(3000)

        input_box = page.locator("textarea")
        count = 0

        for sentence in input_sentences:

            # Clear previous text
            await input_box.click()
            await input_box.fill("")

            # Type naturally
            await input_box.type(sentence + " ", delay=180, timeout=120000)

            await page.wait_for_timeout(350)
            # Read output
            result = await input_box.input_value()

            with open("6_google.txt", "a", encoding="utf-8") as f:
                f.write(result + "\n")

            count += 1
            print(f"{count}: {sentence} -> {result}")

        await browser.close()


asyncio.run(main())
