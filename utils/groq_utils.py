import json
import pandas as pd
from groq import Groq
import requests
from huggingface_hub import InferenceClient
import os
from google.oauth2 import service_account
from google.cloud import texttospeech
from huggingface_hub import InferenceClient
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import ffmpeg

def extract_details_to_dataframe(user_inputs, groq_api_key, user_tone=None, user_platform=None):
    """
    Extracts tone, platform, and topic from a list of user input texts using Groq and saves the results in a DataFrame.

    Args:
        user_inputs (list): List of input texts describing the requirements for the post.
        groq_api_key (str): API key for the Groq service.

    Returns:
        pd.DataFrame: A DataFrame with columns 'Input', 'Tone', 'Platform', 'Topic'.
    """
    # Initialize Groq client
    client = Groq(api_key=groq_api_key)

    results = []
    for user_input in user_inputs:
        try:
            # Meta-prompt for Groq
            meta_prompt = (
                f"Analyze the following text and extract the tone, platform, and topic. "
                f"Only give the results as a JSON object with keys 'tone', 'platform', and 'topic': , not even intro line just data"
                f"'{user_input}'"
            )

            # Send meta-prompt to Groq
            response = client.chat.completions.create(
                messages=[
                    {"role": "user", "content": meta_prompt}
                ],
                model="llama-3.3-70b-versatile",
            )

            # Extract and parse details
            extracted_details = response.choices[0].message.content.strip()
            print("Extracted Details (Raw):", extracted_details)  # Debugging output

            # Parse details safely
            try:
                details = json.loads(extracted_details)  # Parse JSON directly
                tone = details.get("tone", "Error").capitalize()
                platform = details.get("platform", "Error").capitalize()
                topic = details.get("topic", "Error").capitalize()
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                tone = platform = topic = "Error"

            # Add input and parsed details to results
            results.append({
                "Input": user_input,
                "Tone": tone,
                "Platform": platform,
                "Topic": topic,
            })

        except Exception as e:
            print(f"Error processing input: {user_input}\nError: {e}")
            results.append({
                "Input": user_input,
                "Tone": "Error",
                "Platform": "Error",
                "Topic": "Error"
            })

    # Create a DataFrame from results
    df = pd.DataFrame(results)
    return df


def get_bing_news(search_query, BING_API_KEY, freshness="Day", sort_by="Relevance", count=2):
    """
    Fetches news from Bing News Search API based on the provided query and filters.

    Args:
        search_query (str): The topic to search for.
        freshness (str): 'Day', 'Week', or 'Month' to filter results by recency.
        sort_by (str): 'Relevance' or 'Date' to sort results.
        count (int): Number of articles to fetch.

    Returns:
        list: A list of article URLs.
    """
    url = "https://api.bing.microsoft.com/v7.0/news/search"
    headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY}
    params = {
        "q": search_query,
        "freshness": freshness,
        "sortBy": sort_by,
        "count": count,
        "mkt": "en-US"  # Market locale
    }
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    news_data = response.json()
    article_urls = [article["url"] for article in news_data.get("value", [])]
    return article_urls

def fetch_news_for_single_topic_expand_rows(df, BING_API_KEY, topic_column="Topic", freshness="Day", sort_by="Relevance", count=5):
    """
    Fetches the most relevant news articles for a single topic in the DataFrame and expands rows for each article,
    keeping other columns consistent.

    Args:
        df (pd.DataFrame): DataFrame containing a single topic in the specified column, with additional columns.
        topic_column (str): Column name containing the topic.
        freshness (str): 'Day', 'Week', or 'Month' to filter results by recency.
        sort_by (str): 'Relevance' or 'Date' to sort results.
        count (int): Number of articles to fetch for each topic.

    Returns:
        pd.DataFrame: Expanded DataFrame with one row per article URL, retaining other columns.
    """
    if topic_column not in df.columns:
        raise ValueError(f"Column '{topic_column}' not found in the DataFrame.")
    if df[topic_column].isnull().any():
        raise ValueError("The topic column contains null values.")

    # Extract the single row of the original DataFrame
    original_row = df.iloc[0].to_dict()
    topic = original_row[topic_column]

    try:
        print(f"Fetching articles for topic: {topic}")
        article_urls = get_bing_news(topic, BING_API_KEY,  freshness, sort_by, count)

        # Create a new DataFrame with one row per article, duplicating other columns
        expanded_rows = [{**original_row, "Article URL": url} for url in article_urls]
        expanded_df = pd.DataFrame(expanded_rows)

        return expanded_df
    except Exception as e:
        print(f"Error fetching articles for topic '{topic}': {e}")
        return pd.DataFrame(columns=[*df.columns, "Article URL"])  # Return an empty DataFrame
    



def summarize_and_extract_topics(article_urls, groq_api_key):
    """
    Summarizes articles and extracts their main topics using the Groq API.
    
    Args:
        article_urls (list): List of article URLs to process.
        groq_api_key (str): API key for the Groq service.
    
    Returns:
        pd.DataFrame: A DataFrame with columns: 'URL', 'Summary', 'Topic'.
    """
    if not article_urls:
        print("No article URLs provided.")
        return pd.DataFrame(columns=["URL", "Summary", "Topic"])

    # Initialize Groq client
    client = Groq(api_key=groq_api_key)

    results = []
    for url in article_urls:
        try:
            # Step 1: Summarize the article
            summary_response = client.chat.completions.create(
                messages=[
                    {"role": "user", "content": f"Summarize the article at the following URL: {url}"}
                ],
                model="llama-3.3-70b-versatile",
            )
            summary = summary_response.choices[0].message.content.strip()

            # Step 2: Extract the topic from the summary
            topic_response = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"Extract the main topic of this summary in 1-2 words: {summary}. Only provide the topic."
                    }
                ],
                model="llama-3.3-70b-versatile",
            )
            topic = topic_response.choices[0].message.content.strip()

            results.append({"URL": url, "Summary": summary, "Topic": topic})
        except Exception as e:
            print(f"Error processing {url}: {e}")
            results.append({"URL": url, "Summary": "Error generating summary", "Topic": "Error extracting topic"})

    # Convert results to DataFrame
    df = pd.DataFrame(results)
    df.to_csv('articles.csv', index=False)
    return df


def process_articles_with_summaries(articles_df, url_column="Article URL", groq_api_key=None):
    """
    Processes the articles in the DataFrame by summarizing and extracting topics using the Groq API.

    Args:
        articles_df (pd.DataFrame): DataFrame containing the articles to process.
        url_column (str): The name of the column containing the article URLs.
        groq_api_key (str): API key for the Groq service.

    Returns:
        pd.DataFrame: Updated DataFrame with 'Summary' and 'Topic' columns added.
    """
    if url_column not in articles_df.columns:
        raise ValueError(f"Column '{url_column}' not found in the DataFrame.")
    
    # Extract URLs from the DataFrame
    article_urls = articles_df[url_column].dropna().tolist()  # Remove any null URLs

    # Use the provided function to summarize and extract topics
    summarized_df = summarize_and_extract_topics(article_urls, groq_api_key)

    # Merge the summarized data back into the original DataFrame
    merged_df = articles_df.merge(summarized_df, how="left", left_on=url_column, right_on="URL")

    # Drop duplicate 'URL' column to avoid confusion
    merged_df = merged_df.drop(columns=["URL"])

    return merged_df


##---------------------------------------------Prompt Agent-------------------------------------------------------

# Function to generate prompts for processed DataFrame
def generate_prompts_with_video_dependency(processed_df, groq_api_key):
    """
    Adds prompts to the processed DataFrame for all output types: text, image, video visuals, and meme.
    Generates voiceover prompts based on the video visuals prompt.

    Args:
        processed_df (pd.DataFrame): DataFrame containing summaries and other metadata.
        groq_api_key (str): API key for the Groq service.

    Returns:
        pd.DataFrame: Updated DataFrame with new columns for each output type's prompts.
    """
    # Validate required columns
    required_columns = ["Summary", "Tone", "Platform", "Topic_x"]
    for col in required_columns:
        if col not in processed_df.columns:
            raise ValueError(f"Column '{col}' is required in the DataFrame but not found.")

    # Initialize Groq client
    from groq import Groq
    client = Groq(api_key=groq_api_key)

    # Output types
    output_types = ["text", "image", "video_visuals", "video_voiceover", "meme"]
    prompts = {output_type: [] for output_type in output_types}

    for _, row in processed_df.iterrows():
        summary = row["Summary"]
        tone = row["Tone"]
        platform = row["Platform"]
        topic = row["Topic_x"]

        try:
            # Skip rows with missing or empty summaries
            if not summary or pd.isna(summary):
                for output_type in output_types:
                    prompts[output_type].append("No summary available")
                continue

            # Generate prompts for each type
            for output_type in ["text", "image", "video_visuals", "meme"]:
                if output_type == "text":
                    meta_prompt = (
                        f"Generate a concise input prompt that can be fed into an LLM to create a {tone.lower()} post for {platform.lower()} "
                        f"on the topic '{topic}', based on this summary: '{summary}'. Provide only the prompt text, nothing extra words."
                    )
                elif output_type == "image":
                    meta_prompt = (
                        f"Generate a concise input prompt that can be fed into an AI model to create an image for a {tone.lower()} "
                        f"{platform.lower()} post on the topic '{topic}', based on this summary: '{summary}'. Provide only the prompt text, nothing extra words."
                    )
                elif output_type == "video_visuals":
                    meta_prompt = (
                        f"Generate three distinct and concise visual prompts for creating a {tone.lower()} video for a "
                        f"{platform.lower()} post on the topic '{topic}', based on this summary: '{summary}'. "
                        "Each prompt should describe only the visual elements. Return the prompts as three separate lines, "
                        "with no additional words or explanations."
                    )
                elif output_type == "meme":
                    meta_prompt = (
                        f"Generate a concise input prompt that can be fed into an AI model to create a {tone.lower()} meme for a "
                        f"{platform.lower()} post on the topic '{topic}', based on this summary: '{summary}'. Provide only the prompt text, nothing extra words."
                    )

                # Send the meta-prompt to Groq
                response = client.chat.completions.create(
                    messages=[
                        {"role": "user", "content": meta_prompt}
                    ],
                    model="llama-3.3-70b-versatile",
                )
                # Extract the generated prompt
                prompt = response.choices[0].message.content.strip()
                prompts[output_type].append(prompt)

            # Generate voiceover prompt based on the video visuals prompt
            video_visuals_prompt = prompts["video_visuals"][-1]  # Get the last generated video visuals prompt
            meta_prompt_voiceover = (
                f"Based on this video visuals prompt: '{video_visuals_prompt}', generate a 30 words voiceover script that aligns with the visuals "
                f"and matches the {tone.lower()} tone for a {platform.lower()} post on the topic '{topic}'. Provide only the script text, nothing extra words"
            )

            response_voiceover = client.chat.completions.create(
                messages=[
                    {"role": "user", "content": meta_prompt_voiceover}
                ],
                model="llama-3.3-70b-versatile",
            )
            # Extract the generated voiceover script
            voiceover_prompt = response_voiceover.choices[0].message.content.strip()
            prompts["video_voiceover"].append(voiceover_prompt)

        except Exception as e:
            print(f"Error generating prompts for summary: {summary}\nError: {e}")
            for output_type in output_types:
                prompts[output_type].append("Error generating prompt")

    # Add the generated prompts to the DataFrame
    for output_type in output_types:
        processed_df[f"{output_type.replace('_', ' ').capitalize()} Prompt"] = prompts[output_type]

    return processed_df



##------------------------------------------Meme template-----------------------------------------------

def suggest_and_generate_meme_content(
    df, 
    meme_prompt_column="Meme Prompt", 
    tone_column="Tone", 
    platform_column="Platform", 
    topic_column="Topic_x", 
    meme_data_file="meme_data.json", 
    groq_api_key=None
):
    """
    Suggests valid meme templates based on the meme prompts in the DataFrame, validates them against the meme_data.json file, 
    and generates concise meme content based on the valid template and its required lines.

    Args:
        df (pd.DataFrame): DataFrame containing the meme prompts and related metadata.
        meme_prompt_column (str): Column name containing the meme prompts.
        tone_column (str): Column name containing the tone of the content.
        platform_column (str): Column name containing the platform information.
        topic_column (str): Column name containing the topic of the content.
        meme_data_file (str): Path to the JSON file containing meme template information.
        groq_api_key (str): API key for the Groq service.

    Returns:
        pd.DataFrame: Updated DataFrame with new columns 'Meme Template' and 'Meme Content'.
    """
    if meme_prompt_column not in df.columns:
        raise ValueError(f"Column '{meme_prompt_column}' not found in the DataFrame.")
    if tone_column not in df.columns or platform_column not in df.columns or topic_column not in df.columns:
        raise ValueError(f"Columns '{tone_column}', '{platform_column}', or '{topic_column}' not found in the DataFrame.")

    # Load meme template data from JSON
    with open(meme_data_file, "r") as file:
        meme_data = json.load(file)

    # Initialize Groq client
    client = Groq(api_key=groq_api_key)

    meme_templates = []
    meme_contents = []

    for _, row in df.iterrows():
        meme_prompt = row[meme_prompt_column]
        tone = row[tone_column]
        platform = row[platform_column]
        topic = row[topic_column]

        try:
            # Skip rows with missing or empty meme prompts
            if not meme_prompt or pd.isna(meme_prompt):
                meme_templates.append("No template available")
                meme_contents.append("No content generated")
                continue

            valid_template = None
            suggested_template = None

            # Loop until a valid template is found
            while not valid_template:
                # Step 1: Suggest a meme template
                meta_prompt_template = (
                    f"Based on this meme description: '{meme_prompt}', suggest a popular meme template that aligns with the tone '{tone}', "
                    f"the topic '{topic}', and is suitable for a {platform.lower()} post. Provide only the meme template name."
                )
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": meta_prompt_template}],
                    model="llama-3.3-70b-versatile",
                )
                suggested_template = response.choices[0].message.content.strip()

                # Step 2: Validate the suggested template
                valid_template = next(
                    (template for template in meme_data if template["name"].lower() == suggested_template.lower()),
                    None
                )

                if not valid_template:
                    print(f"Suggested template '{suggested_template}' not found in meme_data.json. Retrying...")

            # Save the validated template name
            meme_templates.append(valid_template["name"])

            # Step 3: Generate concise meme content
            lines_required = valid_template["lines"]
            meta_prompt_content = (
                f"Generate a meme content with exactly {lines_required} lines using the '{valid_template['name']}' template. "
                f"The meme should align with this description: '{meme_prompt}', and reflect the {tone.lower()} tone. "
                "Do not include any explanations or descriptions. Provide only the text lines for the meme, separated by newlines."
            )
            response_content = client.chat.completions.create(
                messages=[{"role": "user", "content": meta_prompt_content}],
                model="llama-3.3-70b-versatile",
            )
            meme_content = response_content.choices[0].message.content.strip()
            meme_contents.append(meme_content)

        except Exception as e:
            print(f"Error processing meme prompt: {meme_prompt}\nError: {e}")
            meme_templates.append("Error suggesting template")
            meme_contents.append("Error generating content")

    # Add the suggested meme templates and generated content to the DataFrame
    df["Meme Template"] = meme_templates
    df["Meme Content"] = meme_contents

    return df


#--------------------------------------------Generate Text Posts-----------------------------------------------------------

def generate_text_posts(df, text_prompt_column="Text Prompt", groq_api_key=None):
    """
    Generates text posts based on the text prompts in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the text prompts.
        text_prompt_column (str): Column name containing the text prompts.
        groq_api_key (str): API key for the Groq service.

    Returns:
        pd.DataFrame: Updated DataFrame with a new column 'Generated Text Post' containing the text posts.
    """
    if text_prompt_column not in df.columns:
        raise ValueError(f"Column '{text_prompt_column}' not found in the DataFrame.")

    # Initialize Groq client
    from groq import Groq
    client = Groq(api_key=groq_api_key)

    generated_posts = []

    for _, row in df.iterrows():
        text_prompt = row[text_prompt_column]

        try:
            # Skip rows with missing or empty text prompts
            if not text_prompt or pd.isna(text_prompt):
                generated_posts.append("No text prompt available")
                continue

            # Meta-prompt to generate the text post
            meta_prompt = (
                f"Based on this input prompt: '{text_prompt}', generate a complete text post. "
                "Ensure the tone aligns with the prompt, and provide the post text only."
            )

            # Send the meta-prompt to Groq
            response = client.chat.completions.create(
                messages=[
                    {"role": "user", "content": meta_prompt}
                ],
                model="llama-3.3-70b-versatile",
            )
            # Extract the generated text post
            generated_post = response.choices[0].message.content.strip()
            generated_posts.append(generated_post)

        except Exception as e:
            print(f"Error generating text post for prompt: {text_prompt}\nError: {e}")
            generated_posts.append("Error generating text post")

    # Add the generated text posts to the DataFrame
    df["Generated Text Post"] = generated_posts

    return df


##--------------------------------------------Generate Image Post--------------------------------------------------------------


def generate_images(df, image_prompt_column="Image Prompt", output_dir="generated_images", hf_token=None):
    """
    Generates images using the image prompts in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the image prompts.
        image_prompt_column (str): Column name containing the image prompts.
        output_dir (str): Directory to save the generated images.
        hf_token (str): Hugging Face API token.

    Returns:
        pd.DataFrame: Updated DataFrame with a new column 'Generated Image Path' containing paths to the generated images.
    """
    if image_prompt_column not in df.columns:
        raise ValueError(f"Column '{image_prompt_column}' not found in the DataFrame.")

    if not hf_token:
        raise ValueError("Hugging Face API token (hf_token) is required.")

    # Initialize the Hugging Face client
    client = InferenceClient(token=hf_token)

    output_dir = os.path.join("static", "images", "generated")
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    generated_image_paths = []

    for index, row in df.iterrows():
        image_prompt = row[image_prompt_column]

        try:
            # Skip rows with missing or empty image prompts
            if not image_prompt or pd.isna(image_prompt):
                generated_image_paths.append("No image prompt available")
                continue

            # Generate the image
            print(f"Generating image for prompt: {image_prompt}")
            response = client.text_to_image(model="stabilityai/stable-diffusion-3.5-large", prompt=image_prompt)

            # Save the generated image
            output_file = os.path.join(output_dir, f"generated_image_{index}.png")
            response.save(output_file)  # Use the save method of the PIL Image object
            generated_image_paths.append(output_file)

        except Exception as e:
            print(f"Error generating image for prompt: {image_prompt}\nError: {e}")
            generated_image_paths.append("Error generating image")

    # Add the generated image paths to the DataFrame
    df["Generated Image Path"] = generated_image_paths

    return df




##---------------------------------------------------meme generation agent--------------------------------------------------------------

import requests
import os
import json

def load_meme_template_ids(meme_data_file):
    """
    Load meme template IDs from the JSON file.

    Args:
        meme_data_file (str): Path to the meme_data.json file.

    Returns:
        dict: Mapping of meme template names to their corresponding IDs.
    """
    with open(meme_data_file, "r") as file:
        meme_data = json.load(file)
    return {template["name"]: template["id"] for template in meme_data}

def generate_memes_from_dataframe(
    df, 
    meme_data_file="meme_data.json", 
    template_column="Meme Template", 
    content_column="Meme Content", 
    output_dir = os.path.join("static", "images", "generated")
):
    """
    Generates memes based on a DataFrame containing meme templates and content, using IDs from meme_data.json.

    Args:
        df (pd.DataFrame): DataFrame containing the meme templates and content.
        meme_data_file (str): Path to the meme_data.json file.
        template_column (str): Column name containing the meme template names.
        content_column (str): Column name containing the text content for the memes.
        output_dir (str): Directory to save the generated meme images.

    Returns:
        pd.DataFrame: Updated DataFrame with a new column 'Meme Path' containing paths to the generated memes.
    """
    if template_column not in df.columns:
        raise ValueError(f"Column '{template_column}' not found in the DataFrame.")
    if content_column not in df.columns:
        raise ValueError(f"Column '{content_column}' not found in the DataFrame.")

    # Load meme template IDs from JSON
    meme_template_ids = load_meme_template_ids(meme_data_file)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    meme_paths = []

    for index, row in df.iterrows():
        template_name = row[template_column]
        content = row[content_column]

        try:
            # Validate template name and content
            if not template_name or pd.isna(template_name) or not content or pd.isna(content):
                meme_paths.append("No meme generated")
                continue

            # Map template name to its ID
            template_id = meme_template_ids.get(template_name)
            if not template_id:
                print(f"Template '{template_name}' not found in meme_data.json. Skipping...")
                meme_paths.append("Invalid template")
                continue

            # Split content into lines
            text_lines = [line.strip() for line in content.split("\n") if line.strip()]

            # Validate text line length
            if not text_lines:
                meme_paths.append("No meme generated")
                continue

            # Output file path
            output_file = os.path.join(output_dir, f"meme_{index}.jpg")

            # Generate the meme
            formatted_text = "/".join([text.replace(" ", "_") for text in text_lines])
            meme_url = f"https://api.memegen.link/images/{template_id}/{formatted_text}.jpg"

            # Download and save the meme image
            response = requests.get(meme_url)
            if response.status_code == 200:
                with open(output_file, 'wb') as f:
                    f.write(response.content)
                meme_paths.append(output_file)
                print(f"Meme saved: {output_file}")
            else:
                print(f"Failed to generate meme for row {index}: {response.text}")
                meme_paths.append("Generation failed")
        except Exception as e:
            print(f"Error generating meme for row {index}: {e}")
            meme_paths.append("Error occurred")

    # Add the generated meme paths to the DataFrame
    df["Meme Path"] = meme_paths
    return df

##---------------------------------------------------video generation agent--------------------------------------------------------------

def generate_images_for_video(prompts, hf_token, model_id="stabilityai/stable-diffusion-xl-base-1.0"):
    """
    Generate images from text prompts using a Hugging Face model.

    Args:
        prompts (list): List of text prompts.
        output_dir (str): Directory to save the generated images.
        model_id (str): Hugging Face model ID to use for generation.

    Returns:
        list: List of file paths for the generated images.
    """
    if not hf_token:
        raise ValueError("Hugging Face API token (hf_token) is required.")

    # Initialize the Hugging Face client
    client = InferenceClient(token=hf_token)

    output_dir = os.path.join("static", "videos")
    os.makedirs(output_dir, exist_ok=True)

    image_files = []
    for i, prompt in enumerate(prompts):
        try:
            response = client.text_to_image(model=model_id, prompt=prompt)
            output_file = os.path.join(output_dir, f"image_{i + 1}.png")
            response.save(output_file) 
            image_files.append(output_file)
            print(f"Image {i + 1} saved as '{output_file}'")
        except Exception as e:
            print(f"Error generating image for prompt {prompt}: {e}")
    return image_files

def create_video_from_images(image_files):
    """
    Create a video from a sequence of images.

    Args:
        image_files (list): List of file paths for images.
        output_file (str): Path to save the video.
        fps (int): Frames per second.

    Returns:
        str: Path to the generated video.
    """
    output_dir = os.path.join("static", "videos")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"gen_video.mp4")

    try:
        clip = ImageSequenceClip(image_files, durations=[3.3, 3.3, 3.3]) 
        clip.write_videofile(output_file, codec="libx264")
        print(f"Video saved as '{output_file}'")
        return output_file
    except Exception as e:
        print(f"Error creating video: {e}")

def text_to_speech(text, google_credentials):
    """
    Convert text to speech and save it to an audio file.

    Args:
        text (str): The text to convert to speech.
        output_file (str): The path to save the audio file.

    Returns:
        str: Path to the generated audio file.
    """
    output_dir = os.path.join("static", "videos")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"gen_audio.mp3")

    try:
        # Create credentials from the dictionary
        credentials = service_account.Credentials.from_service_account_info(google_credentials)
        
        # Initialize Text-to-Speech client with credentials
        client = texttospeech.TextToSpeechClient(credentials=credentials)
        
        input_text = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Wavenet-D",
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL,
        )
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        response = client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)
        
        with open(output_file, "wb") as out:
            out.write(response.audio_content)
        print(f"Audio content written to file {output_file}")
        return output_file
    except Exception as e:
        print(f"Error generating audio: {e}")

def combine_video_audio(input_video, input_audio):
    """
    Combines a video file with an audio file into a single video.

    Args:
        input_video (str): Path to the input video file.
        input_audio (str): Path to the input audio file.
        output_video (str): Path to save the output video file.
    """
    try:
        output_dir = os.path.join("static", "videos")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"final_video.mp4")


        # Set up FFmpeg inputs
        video_stream = ffmpeg.input(input_video)
        audio_stream = ffmpeg.input(input_audio)

        # Generate the output video with audio
        (
            ffmpeg
            .output(video_stream, audio_stream, output_file, vcodec='copy', acodec='aac', shortest=None)
            .run(overwrite_output=True, quiet=True)
        )
        print(f"Video with audio created successfully: {output_file}")
        return output_file
    except ffmpeg.Error as e:
        print(f"An FFmpeg error occurred:\n{e.stderr.decode('utf-8')}")
    except Exception as e:
        print(f"An error occurred: {e}")


def generate_video(prompts, narration_text, hf_token, google_credentials):
    # Generate images
    images = generate_images_for_video(prompts, hf_token)

    # Create video from images
    video = create_video_from_images(images)

    # Generate speech audio
    audio = text_to_speech(narration_text, google_credentials)

    # Combine video and audio
    video_path = combine_video_audio(video, audio)
    return video_path
