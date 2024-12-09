# **Minizica - Daily News Content Generator**

Minizica is a revolutionary open-source platform that transforms how you create content. Powered by cutting-edge AI, it effortlessly generates engaging text posts, captivating images, humorous memes, and dynamic videos—all derived from trending news articles. Completely free to use, Minizica is your ultimate tool for saving time, sparking creativity, and producing tailor-made, high-quality content that resonates with your audience. Whether you're a marketer, creator, or just someone with a story to share, Minizica empowers you to stay ahead and make an impact effortlessly!


---

### **Live Demo**

Experience the power of Minizica in action! Explore the live demo and see how effortlessly you can generate high-quality, personalized content from trending news.  

🔗 [Minizica Live Demo](https://daily-news-content-generator.onrender.com)  

Try it now and transform how you create content!  

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

---