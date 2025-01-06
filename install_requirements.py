import subprocess
import sys

def install_requirements():
    """Install required packages from requirements.txt."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    except subprocess.CalledProcessError as e:
        print(f"Failed to install requirements: {e}")
        sys.exit(1)

if __name__ == "__main__":
    install_requirements()

    # Your Flask app import and run goes here
    from process_transcripts import app
    app.run(debug=True)