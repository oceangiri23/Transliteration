from playwright.async_api import async_playwright
import asyncio

input_sentences = [sentences for sentences in open("Transliteration\\Qualitative Analysis\\romanized_sampled.txt", "r", encoding='utf-8').read().splitlines() if sentences != ""]

outputs = []

async def main():

    async with async_playwright() as p:

        browser = await p.chromium.launch(headless=False)

        page = await browser.new_page()

        await page.goto("https://www.easynepalityping.com/")

        await page.wait_for_timeout(3000)

        input_box = page.locator("textarea")
        count = 0

        for sentence in input_sentences:

            # Clear previous text
            await input_box.click()
            await input_box.press("Control+A")
            await input_box.press("Backspace")

            # Type naturally
            await input_box.type(sentence + " ", delay=80)

            await page.wait_for_timeout(100)

            # Read output
            result = await input_box.input_value()

            outputs.append(result)

            count += 1
            print(f"{count}: {sentence} -> {result}")

        await browser.close()

    print(outputs)
    with open("1_google.txt", "w", encoding='utf-8') as f:
        f.write("\n".join(outputs))

asyncio.run(main())