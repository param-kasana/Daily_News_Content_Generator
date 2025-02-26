from flask import Flask, render_template, request, url_for
from utils.groq_utils import extract_details_to_dataframe, generate_modification_prompts_df, generate_text_posts , generate_images , generate_memes_from_dataframe, suggest_and_generate_meme_content, fetch_news_for_single_topic_expand_rows, process_articles_with_summaries, generate_prompts_with_video_dependency, generate_video, get_bing_trending_news, summarize_and_extract_topics
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Access the variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
BING_API_KEY = os.getenv("BING_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
HF_TOKEN_2 = os.getenv("HF_TOKEN_2")
PRIVATE_KEY_ID = os.getenv("PRIVATE_KEY_ID")
PRIVATE_KEY = os.getenv("PRIVATE_KEY")

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
outputs = {}

# Declare global variable
processed_df_with_prompts = None

@app.route('/')
def index():
    # Fetch the trending news URLs
    trend_news_url = get_bing_trending_news(BING_API_KEY, count=5)
    
    # Process the trending news data
    trend_new_df = summarize_and_extract_topics(trend_news_url, GROQ_API_KEY)
    
    # Convert the DataFrame to a dictionary for rendering
    trend_news = trend_new_df.to_dict('records')
    
    # Pass the data to the template
    return render_template('index.html', trend_news=trend_news)

@app.route('/generate', methods=['POST'])
def generate():

    global outputs, processed_df_with_prompts
    outputs.clear()
    
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

    outputs['Summary'] = processed_df_with_prompts['Summary'].iloc[0]
    outputs['Topic_y'] = processed_df_with_prompts['Topic_y'].iloc[0]
    outputs['Article URL'] = processed_df_with_prompts['Article URL'].iloc[0]

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
        meme_data_file = os.path.join("data", "meme_data.json")
        df_with_meme_content = suggest_and_generate_meme_content(
            processed_df_with_prompts, meme_prompt_column="Meme Prompt", tone_column="Tone",
            platform_column="Platform", topic_column="Topic_x", meme_data_file=meme_data_file,
            groq_api_key=GROQ_API_KEY
        )
        df_with_memes = generate_memes_from_dataframe(
            df_with_meme_content, meme_data_file=meme_data_file,
            template_column="Meme Template", content_column="Meme Content"
        )
        outputs['memes'] = [
        meme_path.replace("static/", "") for meme_path in df_with_memes["Meme Path"].tolist()
        ]

    
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
                video_path = generate_video(
                    prompts=video_prompt_list,
                    narration_text=voiceover_prompt,
                    hf_token=HF_TOKEN_2,
                    google_credentials=GOOGLE_CREDENTIALS
                )
                
                video_paths.append(video_path)
            except Exception as e:
                print(f"Error generating video for row {index}: {e}")
                video_paths.append("Error generating video")

        # Add video paths to the outputs
        outputs['videos'] = [
        video_path.replace("static/", "") for video_path in video_paths
        ]

    # Return results to the template
    return render_template('result.html', outputs=outputs)


@app.route('/modify_text', methods=['POST'])
def modify_text():
    existing_text = request.form.get('existing_text')
    new_text = request.form.get('new_text')

    modified_df = generate_modification_prompts_df('text', existing_text, new_text)

    # Logic to regenerate text based on `existing_text` and `new_text`
    processed_df_with_modified_text = generate_text_posts(
            modified_df, text_prompt_column="Text Prompt", groq_api_key=GROQ_API_KEY
        )
    outputs['text'] = processed_df_with_modified_text["Generated Text Post"].tolist()
    return render_template('result.html', outputs=outputs)


@app.route('/modify_image', methods=['POST'])
def modify_image():
    global processed_df_with_prompts  # Access the global DataFrame
    existing_image_prompt = processed_df_with_prompts["Image Prompt"].iloc[0]
    new_image_input = request.form.get('new_image_input')

    modified_df = generate_modification_prompts_df('image', existing_image_prompt, new_image_input)

    processed_df_with_modified_images = generate_images(
            modified_df, image_prompt_column="Image Prompt",
            output_dir = os.path.join("static", "images", "generated"), hf_token=HF_TOKEN
        )
    outputs['images'] = processed_df_with_modified_images["Generated Image Path"].tolist()
    return render_template('result.html', outputs=outputs)


@app.route('/modify_video', methods=['POST'])
def modify_video():
    global processed_df_with_prompts  # Access the global DataFrame
    existing_video_prompt = processed_df_with_prompts["Video visuals Prompt"].iloc[0]
    new_video_input = request.form.get('new_video_input')

    modified_df = generate_modification_prompts_df('video', existing_video_prompt, new_video_input, groq_api_key=GROQ_API_KEY)
    video_paths = []
    output_dir = os.path.join("static", "videos")
    os.makedirs(output_dir, exist_ok=True)

    for index, row in modified_df.iterrows():
        video_prompt = row.get("Video visuals Prompt")
        voiceover_prompt = row.get("Video voiceover Prompt")

        if not video_prompt or not voiceover_prompt:
            video_paths.append("No video generated for this row")
            continue

        try:
            # Convert the 3-line video visuals prompt into a list
            video_prompt_list = video_prompt.splitlines()  # Splits by line breaks into a list

            # Generate video for each row
            video_path = generate_video(
                prompts=video_prompt_list,
                narration_text=voiceover_prompt,
                hf_token=HF_TOKEN,
                google_credentials=GOOGLE_CREDENTIALS
            )
            
            video_paths.append(video_path)
        except Exception as e:
            print(f"Error generating video for row {index}: {e}")
            video_paths.append("Error generating video")

        # Add video paths to the outputs
        outputs['videos'] = [
        video_path.replace("static/", "") for video_path in video_paths
        ]
    return render_template('result.html', outputs=outputs)


@app.route('/modify_meme', methods=['POST'])
def modify_meme():
    global processed_df_with_prompts
    existing_meme_prompt = processed_df_with_prompts["Meme Prompt"].iloc[0]
    new_meme_input = request.form.get('new_meme_input')

    # Logic to regenerate meme based on `existing_meme_prompt` and `new_meme_input`
    modified_df = generate_modification_prompts_df('meme', existing_meme_prompt, new_meme_input)
    modified_df["Tone"] = processed_df_with_prompts["Tone"].iloc[0]
    modified_df["Platform"] = processed_df_with_prompts["Platform"].iloc[0]
    modified_df["Topic_x"] = processed_df_with_prompts["Topic_x"].iloc[0]
    meme_data_file = os.path.join("data", "meme_data.json")
    df_with_meme_content = suggest_and_generate_meme_content(
        modified_df, meme_prompt_column="Meme Prompt", tone_column="Tone",
        platform_column="Platform", topic_column="Topic_x", meme_data_file=meme_data_file,
        groq_api_key=GROQ_API_KEY
    )
    df_with_memes = generate_memes_from_dataframe(
        df_with_meme_content, meme_data_file=meme_data_file,
        template_column="Meme Template", content_column="Meme Content"
    )
    outputs['memes'] = [
    meme_path.replace("static/", "") for meme_path in df_with_memes["Meme Path"].tolist()
    ]
    return render_template('result.html', outputs=outputs)


if __name__ == '__main__':
    app.run(debug=False)
