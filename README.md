# **Minizica - Daily News Content Generator**

Minizica is a revolutionary open-source platform that transforms how you create content. Powered by cutting-edge AI, it effortlessly generates engaging text posts, captivating images, humorous memes, and dynamic videos—all derived from trending news articles. Completely free to use, Minizica is your ultimate tool for saving time, sparking creativity, and producing tailor-made, high-quality content that resonates with your audience. Whether you're a marketer, creator, or just someone with a story to share, Minizica empowers you to stay ahead and make an impact effortlessly!

---
**🏆 Hackathon Champion**  
Minizica proudly secured **1st place** among 200 teams worldwide at the **#BuildwithAI Hackathon 2024**, hosted by **[GenAI Works](https://genai.works)**—the world’s largest AI hub with over 10 million enthusiasts! 🚀🎉  

This annual global competition brought together 4,500 participants to push the boundaries of AI innovation, and Minizica emerged as the top project, demonstrating its potential to revolutionize AI-powered content creation.  

---

### **Watch Minizica in Action**

Check out our demo video to see how Minizica works:  

[![Watch the Demo Video](https://img.youtube.com/vi/YNEF4pFb4BU/0.jpg)](https://www.youtube.com/watch?v=YNEF4pFb4BU)  

Click the image above or [watch the demo video here](https://www.youtube.com/watch?v=YNEF4pFb4BU).

---

### **Explore the Website**

You can explore Minizica's website and see how effortlessly you can generate high-quality, personalized content from trending news.  

🔗 [Explore Minizica](https://daily-news-content-generator.onrender.com)  

Try it now and transform how you create content!  

***Disclaimer:*** This project is in its MVP (Minimum Viable Product) phase, and as such, the website may take some time to start and generate content due to current resource constraints. However, the platform remains completely free to use, ensuring accessible AI-powered content generation for everyone. We appreciate your patience and support as we continue to improve and enhance Minizica!   

---

### **Why Minizica Stands Out**

Minizica is a one-of-a-kind platform that combines multiple AI-driven content generation services—text posts, images, memes, and videos—into a seamless and **completely free** solution.  

Unlike other tools that focus on a single content format or charge high subscription fees, Minizica offers:  
- **All-in-One Solution**: Generate text, images, memes, and videos from one platform.  
- **Highly Customizable**: Tailor content tone, style, and platform to your needs.  
- **Open Source and Free**: Accessible to everyone, with no hidden costs.  
- **Social Media Integration (In Progress)**: Work is underway to enable direct posting to social media platforms and automated scheduling, further simplifying the content-sharing process.  

No other platform provides this range of services in a unified package, making Minizica a **game-changer** for effortless, AI-powered content creation.  

---

### **Features Overview**

- Automatically retrieves relevant news articles.
- Summarizes and extracts key insights using LLMs.
- Generates platform-specific content in multiple formats: text, images, memes, and videos.
- Customizable tone and style based on user inputs (e.g., humorous, formal).
- Simple, responsive web interface for real-time previews and downloads.

---

### **Setup**

#### **Prerequisites**

- **Python**: Version 3.8+
- **Libraries**: Install dependencies from `requirements.txt`.
- **API Keys**:
  - Bing News API
  - Groq API
  - Hugging Face API
- **Environment Variables**: Create a `.env` file for securely storing API keys.

---

#### **Installation**

1. **Clone the Repository**:  
   ```bash
   git clone https://github.com/param-kasana/Daily_News_Content_Generator.git
   cd Daily_News_Content_Generator
   ```  

2. **Install Dependencies**:  
   ```bash
   pip install -r requirements.txt
   ```  

3. **Add API Keys**:  
   Create a `.env` file in the root directory and add:  
   ```plaintext
   GROQ_API_KEY=<your-groq-api-key>
   BING_API_KEY=<your-bing-api-key>
   HF_TOKEN=<your-hugging-face-token>
   ```  

---

#### **Running the App**

1. **Start the Flask Server**:  
   ```bash
   python app.py
   ```  

2. **Access the App**:  
   Open your browser and go to `http://127.0.0.1:5000`.

---

### **Quick Start**

1. **Provide Inputs**:  
   - Specify a topic (e.g., "Climate Change Updates"), tone (e.g., "Informative"), and platform (e.g., "LinkedIn") or provide a detailed request such as "Generate an informative post about renewable energy advancements for a LinkedIn audience."  
   - Choose the content formats you want to generate (e.g., text, images, memes, videos).   

2. **Generate Content**:  
   - Click "Generate" to process the inputs.  

3. **Preview and Download**:  
   - View the generated content in the web interface or refine outputs through follow-up prompts.  
   - Download and use the content for your social media or marketing needs.  

---

Minizica simplifies content creation with powerful AI capabilities, making it accessible and free for everyone. Get started now and transform how you engage with your audience!  

---

### **Detailed Documentation**

For a comprehensive understanding of the project, including its architecture, workflow, additional features, demo screenshots, and future
enhancements refer to the [Detailed Project Report](PROJECT_DETAILS.md).  
