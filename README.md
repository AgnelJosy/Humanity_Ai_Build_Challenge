# Class Quality Analyzer

A Streamlit application that assesses class-recording transcripts against a five-tier teaching-quality rubric using Google Gemini or OpenAI.

## Local setup

Use Python 3.10 or newer (Python 3.11+ is recommended for current Google Gen AI SDK support).

1. Create and activate a virtual environment in this directory:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies and run the app:

   ```powershell
   python -m pip install -r requirements.txt
   streamlit run app.py
   ```

3. Open the local URL printed by Streamlit, typically <http://localhost:8501>.
4. Choose a provider, enter its API key in the password field, and use **Test API connection** before evaluating a transcript. The Gemini default is `gemini-3.8-flash`; choose a model enabled for your project.

## API keys

For local testing, enter a key in the app's password-type field. The app keeps that value in Streamlit session state for the current session and does not intentionally persist it. Never hardcode a key in source code, include it in screenshots, commit it, or paste it into chat.

Alternatively, set `GEMINI_API_KEY` or `OPENAI_API_KEY` in the environment before launching Streamlit. A `.streamlit/secrets.toml` file can also be used for local development, but it must be excluded from version control. Revoke and replace any key that has been exposed. API usage may incur costs or be subject to provider quotas and rate limits.

## Deployment

- Deploy through Streamlit Community Cloud or another trusted hosting service.
- Configure provider keys in the host's secret manager; do not add secrets to the repository or build logs.
- Use a supported, maintained Python release and install dependencies from `requirements.txt` during deployment.
- Restrict access to the app appropriately. Transcripts may contain personal or sensitive information; obtain authorization before uploading and follow applicable retention and privacy requirements.
- The app sends transcript text to the selected external AI provider for processing. Gemini requests set `store` to false; provider policies and service behavior may change, so review current terms before deployment.
- Review `app_errors.log` access and retention. It records operational error details; protect or clear it according to your organization's policy.

## Features and limitations

- Upload `.txt`/`.vtt` transcripts or paste text directly.
- Evaluates interaction, introductions, homework, conclusions, clarity, and related rubric items.
- Visual criteria (background, camera, face visibility) cannot be verified from speech alone. They remain `unclear` unless the transcript contains an explicit spoken cue.
- Model assessments are probabilistic and can misread noisy speech-to-text, names, or context. Review the transcript and results before using them for decisions.
- The optional OpenAI provider requires a valid OpenAI API key and its SDK; provider billing and model access are separate from Gemini.

## Errors and logs

Application errors are recorded in `app_errors.log` and surfaced in the Streamlit UI as an error dialog where supported, with an inline fallback for older Streamlit versions. Do not include API keys in error reports.