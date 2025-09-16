# 🌱 End-to-End Plant Village Project  

In this project, we are working with the **PlantVillage dataset**.  
The problem we are trying to solve is **early detection of plant diseases**. Farmers should be able to quickly identify whether their plants are vulnerable to a disease and take action as soon as possible.  

We will train a **Deep Learning model (CNN)** and deploy it into a web application.  
The application will allow farmers to either:  
- 📤 Upload a recent photo of their plant, or  
- 📸 Use a live camera to capture the plant’s image.  

The system will then instantly respond with:  
1. ✅ Whether the plant is healthy or diseased.  
2. 🧾 If diseased, the type of disease detected.  
3. 🌿 Recommended next steps for the farmer.  

---

## 🔧 How We Are Solving the Problem  

- Using **Convolutional Neural Networks (CNNs)** to classify plant images.  
- Collecting and cleaning a high-quality **PlantVillage dataset from Kaggle**.  
- Building a complete **Machine Learning pipeline** for training, evaluating, and deploying the model.  
- Creating a **user-friendly web interface** where farmers can easily interact with the system.  

---

## 📊 ML Pipeline Workflow  

1. **Data Ingestion** – Collect and load data.    
2. **Model Training** – Train a CNN model on plant disease classification.  
3. **Model Evaluation** – Measure model accuracy and performance.  
4. **Deployment** – Create a web app for real-time predictions.  

---

## 🛠️ Project Workflows  

1. Update `config.yaml`  
2. Update `schema.yaml`  
3. Update `params.yaml`  
4. Update the entity classes  
5. Update the configuration manager inside `config/`  
6. Update the components  
7. Update the pipeline  
8. Update `main.py`  
9. Update `templates.py`  
10. Update `app.py`  

---

## 🚀 Features  

- Real-time plant disease detection  
- Upload image 
- Instant feedback with health status  
- End-to-end ML pipeline with MLOps practices  

---

## 💻 Tech Stack  

- **Python** 🐍  
- **Pytorch** for Deep Learning  
- **Flask / FastAPI** for Web App  
- **MLflow / DVC** for Experiment Tracking  
- **Kaggle Dataset (PlantVillage)** 🌱  

---

✅ This project combines **Deep Learning, MLOps practices, and Web Development** to deliver a real-world solution that can help farmers protect their crops and improve yields.  
