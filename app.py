from flask import Flask, render_template, request, url_for
from utils.groq_utils import extract_details_to_dataframe, generate_text_posts , generate_images , generate_memes_from_dataframe, suggest_and_generate_meme_content, fetch_news_for_single_topic_expand_rows, process_articles_with_summaries, generate_prompts_with_video_dependency, generate_video
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Get GROQ API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
BING_API_KEY = os.getenv("BING_API_KEY")

# Hugging Face Token
HF_TOKEN = os.getenv('HF_TOKEN')

# Extract the sensitive key from the environment
PRIVATE_KEY = os.getenv("GOOGLE_PRIVATE_KEY")  # Your private key from the JSON
PRIVATE_KEY_ID = os.getenv("GOOGLE_PRIVATE_KEY_ID")  # Your private key ID

# Construct the credentials dictionary dynamically
GOOGLE_CREDENTIALS = {
    "type": "service_account",
    "project_id": "gen-lang-client-0264847684",
    "private_key_id": PRIVATE_KEY_ID,
    "private_key": PRIVATE_KEY,
    "client_email": "text-to-speech-service-account@gen-lang-client-0264847684.iam.gserviceaccount.com",
    "client_id": "115310702587466426192",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/text-to-speech-service-account%40gen-lang-client-0264847684.iam.gserviceaccount.com",
    "universe_domain": "googleapis.com",
}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    # Step 1: Get user inputs
    prompt = request.form.get('prompt')
    tone = request.form.get('tone')
    platform = request.form.get('platform')
    content_types = request.form.getlist('content_types')  # Selected content types

    # Step 2: Extract details from the user input
    user_inputs = [prompt]
    df = extract_details_to_dataframe(user_inputs, GROQ_API_KEY, user_tone=tone, user_platform=platform)

    if df.empty:
        return render_template('result.html', error="No details extracted from the input.")

    # Step 3: Fetch related news articles
    articles_df = fetch_news_for_single_topic_expand_rows(
        df, BING_API_KEY, topic_column="Topic", freshness="Day", sort_by="Relevance", count=1
    )

    if articles_df.empty:
        return render_template('result.html', error="No news articles found for the topic.")

    # Step 4: Summarize articles and extract topics
    processed_df = process_articles_with_summaries(
        articles_df, url_column="Article URL", groq_api_key=GROQ_API_KEY
    )

    # Step 5: Generate prompts for all output types
    processed_df_with_prompts = generate_prompts_with_video_dependency(
        processed_df, groq_api_key=GROQ_API_KEY
    )
    processed_df_with_prompts.to_csv("processed_df_with_prompts.csv", index=False)
    # Step 6: Generate content based on user selections
    outputs = {}

    # Generate Text Posts
    if 'text' in content_types:
        processed_df_with_text = generate_text_posts(
            processed_df_with_prompts, text_prompt_column="Text Prompt", groq_api_key=GROQ_API_KEY
        )
        outputs['text'] = processed_df_with_text["Generated Text Post"].tolist()

    # Generate Images
    if 'image' in content_types:
        processed_df_with_images = generate_images(
            processed_df_with_prompts, image_prompt_column="Image Prompt",
            output_dir = os.path.join("static", "images", "generated"), hf_token=HF_TOKEN
        )
        outputs['images'] = processed_df_with_images["Generated Image Path"].tolist()

    # Generate Memes
    if 'meme' in content_types:
        df_with_meme_content = suggest_and_generate_meme_content(
            processed_df_with_prompts, meme_prompt_column="Meme Prompt", tone_column="Tone",
            platform_column="Platform", topic_column="Topic_x", meme_data_file="meme_data.json",
            groq_api_key=GROQ_API_KEY
        )
        df_with_memes = generate_memes_from_dataframe(
            df_with_meme_content, meme_data_file="meme_data.json",
            template_column="Meme Template", content_column="Meme Content"
        )
        outputs['memes'] = df_with_memes["Meme Path"].tolist()

    
    # Step 6: Generate Videos
    if 'video' in content_types:
        video_paths = []
        output_dir = os.path.join("static", "videos")
        os.makedirs(output_dir, exist_ok=True)

        for index, row in processed_df_with_prompts.iterrows():
            video_prompt = row.get("Video visuals Prompt")
            voiceover_prompt = row.get("Video voiceover Prompt")

            if not video_prompt or not voiceover_prompt:
                video_paths.append("No video generated for this row")
                continue

            try:
                # Convert the 3-line video visuals prompt into a list
                video_prompt_list = video_prompt.splitlines()  # Splits by line breaks into a list

                # Generate video for each row
                video_file = generate_video(
                    prompts=video_prompt_list,
                    narration_text=voiceover_prompt,
                    hf_token=HF_TOKEN,
                    google_credentials=GOOGLE_CREDENTIALS
                )
                # Save video to the output directory
                video_path = os.path.join(output_dir, f"generated_video_{index}.mp4")
                with open(video_path, 'wb') as f:
                    f.write(video_file)

                video_paths.append(video_path)
            except Exception as e:
                print(f"Error generating video for row {index}: {e}")
                video_paths.append("Error generating video")

        # Add video paths to the outputs
        outputs['videos'] = video_paths
    
    # # Generate Video
    # if 'video' in content_types:
    #     # Example usage
    #     prompts = [
    #         "A cartoon of a robot performing stand-up comedy while a human looks nervous in the audience.",
    #         "A robot telling jokes on stage with a spotlight while a human laughs nervously.",
    #         "A humanoid robot delivering a comedy monologue, audience looking skeptical.",
    #     ]
        
    #     narration_text = "Welcome to the stand-up comedy night with our star performer, the humanoid robot! Welcome to the stand-up comedy night with our star performer, the humanoid robot!"

    #     generate_video(prompts, narration_text, hf_token=HF_TOKEN, google_credentials = GOOGLE_CREDENTIALS)


    # Return results to the template
    return render_template('result.html', outputs=outputs)


if __name__ == '__main__':
    app.run(debug=False)
