import re

def check_unclosed_braces(file_path):
    """
    Reads a file, extracts JavaScript code blocks, and checks for unclosed braces.
    """
    with open(file_path, 'r') as file:
        content = file.read()

    # Extract all JavaScript blocks (inside <script> tags or inline JavaScript in <script> tags)
    js_blocks = re.findall(r'<script.*?>(.*?)</script>', content, re.DOTALL)

    if not js_blocks:
        print("No JavaScript code found.")
        return

    for idx, block in enumerate(js_blocks, start=1):
        print(f"Checking JavaScript block {idx}...")

        # Count opening and closing braces
        open_braces = block.count('{')
        close_braces = block.count('}')

        if open_braces != close_braces:
            print(f"Unclosed braces found in JavaScript block {idx}:")
            print(f"Opening braces: {open_braces}, Closing braces: {close_braces}")
        else:
            print(f"JavaScript block {idx} is balanced.")

if __name__ == "__main__":
    # Replace 'your_file.html' with the actual path to your file
    file_path = 'templates/query_results.html'  # Update with your HTML file name
    check_unclosed_braces(file_path)