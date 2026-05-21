from playwright.sync_api import sync_playwright

with open("romanized.txt", "r", encoding='utf-8') as f:
    input_sentences = f.read().splitlines()
    input_sentences = [sentence for sentence in input_sentences if sentence != ""]


outputs = []

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)

    page = browser.new_page()
    page.goto("https://huggingface.co/spaces/Sagar32/Romanized-To-Devanagari-Transliteration")

    page.wait_for_timeout(3000)

    input_box = page.locator("textarea")
    count = 0

    for sentence in input_sentences:
        count += 1

        # Clear textbox
        input_box.click()
        input_box.press("Control+A")
        input_box.press("Backspace")

        # Type like real keyboard
        input_box.type(sentence + " ", delay=40)

        # Wait for transliteration
        page.wait_for_timeout(100)

        # Get converted text
        result = input_box.input_value()

        outputs.append(result)

        print(f"{count}: {sentence} -> {result}")

    browser.close()
with open("1_google.txt", "w", encoding='utf-8') as f:
    f.write("\n".join(outputs))

print(outputs)