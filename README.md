
# Federated Learning for Robust DDoS Detection

This project evaluates the robustness of Federated Learning (FL) against **label-flipping adversarial attacks** in Distributed Denial-of-Service (DDoS) detection. It compares a neural network-based FL framework with traditional centralized models: Decision Tree (DT), Bagging Decision Tree (DTB), and Deep Neural Network (DNN), using the [UNSW-NB15 dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset).

## 🧠 Key Features

- Federated Learning framework using the [Flower](https://flower.dev/) library.
- Evaluation under adversarial poisoning scenarios (5%, 10%, 15% label flipping).
- Comparison with centralized models: DT, DTB, and DNN.
- Robustness testing using standard metrics: Accuracy, Precision, Recall, F1-Score.
- Implementation structured into modular scripts for client, server, and task handling.

---

## 📁 Project Structure

```
.
├── client_app.py         # FL client logic
├── server_app.py         # FL server orchestration with FedAvg strategy
├── task.py               # Shared model definition and evaluation utilities
├── DecisionTree.ipynb    # Centralized Decision Tree and Bagging models
├── DNN_base_model.ipynb  # Centralized Deep Neural Network baseline
├── Rohan_Project-5.pdf   # Research report describing the methodology and results
```

---

## 🏗️ Requirements

- Python ≥ 3.9
- Flower (Federated Learning framework)
- PyTorch
- NumPy, Pandas, Scikit-learn
- Matplotlib (optional, for visualization)

Install the dependencies:
```bash
pip install -r requirements.txt
```

Suggested `requirements.txt`:
```text
flwr==1.4.0
torch>=1.12
numpy
pandas
scikit-learn
matplotlib
jupyter
```

---

## ⚙️ How to Run

### Step 1: Centralized Baselines
Run the following notebooks locally:
- `DecisionTree.ipynb`: Builds and evaluates DT and DTB models.
- `DNN_base_model.ipynb`: Trains a standalone deep neural network.

### Step 2: Federated Learning
Open two terminals and execute the following:

#### Terminal 1: Start the FL Server
```bash
python server_app.py
```

#### Terminal 2: Start Multiple Clients (in separate terminals or loop)
```bash
python client_app.py
```

Each client trains locally on poisoned/non-poisoned data and sends updates to the server using **Federated Averaging**.

---

## 📊 Evaluation

The models are tested under varying levels of adversarial label flipping:
- **0% (Clean)**
- **5%, 10%, 15% (Poisoned)**

### Metrics Used:
- Accuracy
- Precision
- Recall
- F1-Score

### Observations:
- **DTB** performed best under clean conditions.
- **FL** showed **higher recall** and **lower drop in recall** under poisoned settings—suggesting **greater resilience**.

| Poisoning | Model | Accuracy | Precision | Recall | F1-Score |
|-----------|--------|----------|-----------|--------|----------|
| 0%        | DTB    | 92.95%   | 0.9311    | 0.9295 | 0.9294   |
| 15%       | FL     | 80.96%   | 0.7941    | 0.8355 | 0.8142   |

For full results and analysis, refer to `Rohan_Project-5.pdf`.

---

## 📌 Future Work

- Introduce **non-IID data distribution** among FL clients.
- Incorporate **robust aggregation techniques** (e.g., Multi-Krum, Trimmed Mean).
- Use **anomaly detection** for poisoned updates.
- Integrate **Explainable AI (XAI)** into the FL pipeline.

---

## 📖 Citation

If you use this code or work in your research, please cite:

```
@article{soares2025flpoison,
  title={Evaluating Federated Learning Resilience Against Adversarial Data Poisoning in DDoS Detection},
  author={Soares, Rohan and Fiddler, Liam and Simpson, Zak and Saha, Sajal},
  institution={University of Northern British Columbia},
  year={2025}
}
```

---

## 🤝 Contributors

- Rohan Soares  
- Liam Fiddler  
- Zak Simpson  
- Sajal Saha, Member IEEE

---

## 📬 Contact

For questions or collaboration, reach out to [saha2@unbc.ca](mailto:saha2@unbc.ca).
