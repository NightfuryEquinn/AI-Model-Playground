import logging
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from transformers import pipeline

# Suppress transformers warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

class QuantumChatbot:
  def __init__(self, max_length: int = 100):
    # Initialise components once
    self.gpt_pipeline = pipeline(
      task = "text-generation",
      model = "gpt2",
      device = 0 if self._cuda_available() else -1, # Use GPU if available
      torch_dtype = "auto"
    )
    self.backend = Aer.get_backend('statevector_simulator')
    self.max_length = max_length

  @staticmethod
  def _cuda_available() -> bool:
    try:
      import torch
      return torch.cuda.is_available()
    except ImportError:
      return False
    
  def text_to_quantum_features(self, text: str, num_features: int = 8) -> np.ndarray:
    """Optimised text encoding with padding and normalisation"""
    # Pad or truncate to exact length
    text_bytes = text.encode('utf-8')[:num_features]
    features = np.zeros(num_features, dtype = np.float32)
    features[:len(text_bytes)] = list(text_bytes)

    # Avoid division by zero
    norm = np.linalg.norm(features)
    if norm > 0:
      features /= norm

    return features
  
  def create_quantum_circuit(self, features: np.ndarray) -> QuantumCircuit:
    """Create and compile quantum circuit once"""
    qc = QuantumCircuit(len(features))

    # Vectorized rotation gates
    for i, val in enumerate(features):
      if val != 0: # Skip zero rotations
        qc.ry(val * np.pi, i)

    # Pre-compile circuit
    return transpile(qc, self.backend, optimization_level = 3)
  
  def get_quantum_signature(self, qc: QuantumCircuit) -> list:
    """Extract quantum signature efficiently"""
    job = self.backend.run(qc, shots = 1)
    result = job.result()
    statevector = result.get_statevector()

    # Return top 5 amplitudes (rounded for consistency)
    amplitudes = np.abs(statevector)
    top_indices = np.argsort(amplitudes)[-5:]

    return [round(amplitudes[i], 4) for i in reversed(top_indices)]
  
  def generate_response(self, user_input: str, quantum_signature: list) -> str:
    # More concise prompt
    enhanced_prompt = f"Quantum analysis: {quantum_signature}. User: {user_input}"

    response = self.gpt_pipeline(
      enhanced_prompt,
      max_length = len(enhanced_prompt) + self.max_length,
      num_return_sequences = 1,
      truncation = True,
      do_sample = True,
      temperature = 0.7,
      pad_token_id = self.gpt_pipeline.tokenizer.eos_token_id
    )[0]['generated_text']

    # Extract only the new generated part
    return response[len(enhanced_prompt):].strip()
  
  def process_input(self, user_input: str) -> str:
    """Process user input through quantum enhancement"""
    # Quantum processing pipeline
    features = self.text_to_quantum_features(user_input)
    qc = self.create_quantum_circuit(features)
    quantum_signature = self.get_quantum_signature(qc)

    return self.generate_response(user_input, quantum_signature)
  
  def run(self):
    """Main chatbot loop with error handling"""
    print("Quantum chatbot initialised (Type 'exit' to quit)")
    print("="*50)

    try:
      while True:
        user_input = input("\n You: ").strip()

        if user_input.lower() in {'exit', 'quit', 'bye'}:
          print('Quantum: Goodbye!')
          break

        if not user_input:
          continue

        try:
          response = self.process_input(user_input)
          print(f"Quantum: {response}")

        except Exception as e:
          print(f'\nError processing input: {e}')
          print('Please try again...')
    
    except KeyboardInterrupt:
      print('\n\nQuantum: Session interrupted. Goodbye!')

if __name__ == '__main__':
  chatbot = QuantumChatbot(max_length = 100)
  chatbot.run()