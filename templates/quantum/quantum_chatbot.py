from transformers import pipeline
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
import numpy as np

# Load GPT-2 Chat Model
gpt_pipeline = pipeline("text-generation", model = "gpt2")

# Generate GPT response
max_prompt_length = 200
def generate_res(prompt, max_length = max_prompt_length):
  res = gpt_pipeline(prompt, max_length = max_length, num_return_sequences = 1, truncation = True)

  return res[0]['generated_text']

# Convert text to quantum features
def text_to_quantum_features(text):
  """Convert text to numerical feature vector using ASCII encoding."""
  features = np.array([ord(c) for c in text[:8]]) # Take first 8 characters
  features = features / np.linalg.norm(features) # Normalize

  return features

# Encode into Quantum Circuit
def quantum_feature_encoding(features):
  num_qubits = len(features)
  qc = QuantumCircuit(num_qubits)

  for i, val in enumerate(features):
    qc.ry(val * np.pi, i) # Encode values into quantum states

  return qc

# Run Quantum Computation
def run_quantum_circuit(qc):
  backend = Aer.get_backend('statevector_simulator')

  compiled_circuit = transpile(qc, backend)
  job = backend.run(compiled_circuit)
  result = job.result()

  statevector = np.array(result.get_statevector())

  return statevector

# Chatbot Loop
def quantum_chatbot():
  print("Quantum Chatbot (Type 'exit' to stop)\n")

  while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
      print("Chatbot: Goodbye!")
      break

    # Convert input to quantum features
    features = text_to_quantum_features(user_input)
    qc = quantum_feature_encoding(features)

    # Run quantum processing
    quantum_result = run_quantum_circuit(qc)

    # Generate GPT response (using quantum-enhanced input)
    quantum_text = f"The quantum system analyzed this input and returned key insights: {[round(abs(c), 4) for c in quantum_result[:5]]}. Based on this, respond to the user naturally in {max_prompt_length} words." # Use part of quantum state
    bot_response = generate_res(quantum_text + " " + user_input)

    print("\nChatbot: ", bot_response, "\n")

# Run chatbot
quantum_chatbot()