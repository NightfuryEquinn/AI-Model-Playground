import logging
from typing import List
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from transformers import pipeline

# Suppress transformers warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

class QuantumMemory:
  """Quantum-inspired memory system for conversation context"""
  def __init__(self, memory_size: int = 8):
    self.memory_size = memory_size
    self.memory_state = np.empty(2 ** memory_size, dtype = np.complex128)
    self.memory_state.fill(0)
    self.memory_state[0] = 1.0 # Initialize to |000...0> state
    self.conversation_history = []

  def encode_text_to_state(self, text: str) -> np.ndarray:
    """Convert text to quantum state representation"""
    # Hash text to get consistent state indices
    text_hash = abs(hash(text) % (2 ** self.memory_size))
    state = np.zeros(2 ** self.memory_size, dtype = np.complex128)

    # Create superposition of related states
    for i in range(min(4, len(text.split()))): # Up to 4-word-based states
      word_hash = hash(text.split()[i]) % (2 ** self.memory_size) if i < len(text.split()) else text_hash
      state[word_hash] += 1.0 / np.sqrt(min(4, len(text.split())))

    # Normalise
    norm = np.linalg.norm(state)
    return state / norm if norm > 0 else state

  def update_memory(self, user_input: str, response: str):
    """Update quantum memory with new conversation turn"""
    # Encode current interaction
    input_state = self.encode_text_to_state(user_input)
    response_state = self.encode_text_to_state(response)

    # Create entangled state (conversation coherence)
    interaction_state = (input_state + response_state) / np.sqrt(2)

    # Quantum interference with existing memory (decay factor 0.7)
    self.memory_state = 0.7 * self.memory_state + 0.3 * interaction_state

    # Normalise to maintain quantum property
    norm = np.linalg.norm(self.memory_state)
    if norm > 1e-10:
      self.memory_state /= norm

    # Keep text history for context (limited to 5 turns)
    self.conversation_history.append((user_input, response))
    if (len(self.conversation_history) > 5):
      self.conversation_history.pop(0)
    
  def get_context_signature(self) -> List[float]:
    """Extract quantum signature representing conversation context"""
    amplitudes = np.abs(self.memory_state)
    top_indices = np.argsort(amplitudes)[-3:] # Top 3 memory states

    return [round(amplitudes[i], 4) for i in reversed(top_indices)]
  
class QuantumChatbot:
  def __init__(self, max_length: int = 100):
    # Initialise components once
    self.gpt_pipeline = pipeline(
      task = 'text-generation',
      model = 'gpt2',
      device = -1,
      torch_dtype = 'auto'
    )

    self.backend = Aer.get_backend('statevector_simulator')
    self.max_length = max_length
    self.quantum_memory = QuantumMemory()

  @staticmethod
  def _cuda_available() -> bool:
    try :
      import torch
      return torch.cuda.is_available()
    except ImportError:
      return False

  def quantum_enhanced_embedding(self, text: str, num_features: int = 8) -> np.ndarray:
    """Create quantum-inspired text embeddings with relationship encoding"""
    # Base features from text bytes
    text_bytes = text.lower().encode('utf-8')[:num_features // 2]
    base_features = np.zeros(num_features // 2, dtype = np.float32)
    base_features[:len(text_bytes)] = list(text_bytes)

    # Normalise base features
    norm = np.linalg.norm(base_features)
    if norm > 0:
      base_features /= norm

    # Create entangled features (relationships between characters / words)
    entangled_features = np.zeros(num_features, dtype = np.float32)
  
    for i in range(num_features // 2):
      # Real part: Original feature
      entangled_features[2 * i] = base_features[i]

      # Imaginary part: Relationship to context (Running average)
      if i > 0:
        entangled_features[2 * i + 1] = np.mean(base_features[:i + 1])
      else:
        entangled_features[2 * i + 1] = base_features[i] * 0.5 
    
    return entangled_features

  def create_quantum_circuit(self, features: np.ndarray) -> QuantumCircuit:
    """Create quantum circuit with enhanced entanglement patterns"""
    qc = QuantumCircuit(len(features))

    # Apply rotation gates
    for i, val in enumerate(features):
      if val != 0:
        qc.ry(val * np.pi, i)

    # Add entanglement gates for feature relationships
    for i in range(0, len(features) - 1, 2):
      if features[i] != 0 and features[i + 1] != 0:
        qc.cx(i, i + 1) # Create entanglement between related features

    return transpile(qc, self.backend, optimization_level = 3)

  def get_quantum_signature(self, qc: QuantumCircuit) -> List[float]:
    """Extract quantum signature with improved measurement"""
    job = self.backend.run(qc, shots = 1)
    result = job.result()
    statevector = result.get_statevector()
    
    # Get probability amplitudes
    amplitudes = np.abs(statevector)
    
    # Focus on top states that represent text features
    top_indices = np.argsort(amplitudes)[-4:]
    signature = [round(amplitudes[i], 4) for i in reversed(top_indices)]
    
    return signature

  def calculate_response_quality(self, response: str, user_input: str) -> float:
    """Calculate response quality score for quantum selection"""
    if not response.strip():
        return 0.1
    
    # Length penalty (too short or too long)
    length_score = max(0.1, 1.0 - abs(len(response.split()) - 15) / 20.0)
    
    # Relevance score (simple word overlap)
    user_words = set(user_input.lower().split())
    response_words = set(response.lower().split())
    relevance_score = len(user_words & response_words) / max(len(user_words), 1) * 0.5 + 0.5
    
    # Coherence score (avoid repetition)
    unique_words = len(set(response.lower().split()))
    total_words = len(response.split())
    coherence_score = unique_words / max(total_words, 1) if total_words > 0 else 0.1
    
    return (length_score + relevance_score + coherence_score) / 3.0

  def generate_superposed_responses(self, user_input: str, quantum_signature: List[float], context_signature: List[float]) -> str:
    """Generate multiple response candidates and select using quantum measurement"""
    
    # Create quantum-enhanced prompts
    context_info = f"Context: {context_signature}" if any(sig > 0.1 for sig in context_signature) else ""
    quantum_info = f"Analysis: {quantum_signature[:2]}"
    
    prompts = [
      f"{context_info} Human: {user_input}\nAssistant:",
      f"Conversation context. {quantum_info}\nHuman: {user_input}\nAssistant:",
      f"Previous context considered.\nHuman: {user_input}\nAssistant:"
    ]
    
    responses = []
    weights = []
    
    for prompt in prompts:
      try:
        result = self.gpt_pipeline(
          prompt,
          max_length = len(prompt) + self.max_length,
          num_return_sequences = 1,
          do_sample = True,
          temperature = 0.8,
          repetition_penalty = 1.1,
          pad_token_id = self.gpt_pipeline.tokenizer.eos_token_id
        )[0]['generated_text']
        
        response = result[len(prompt):].strip()
        
        if response:
          responses.append(response)
          quality = self.calculate_response_quality(response, user_input)
          weights.append(quality)
          
      except Exception as e:
        continue
    
    if not responses:
      return "I'm having trouble processing that. Could you rephrase?"
    
    # Quantum measurement: probabilistic selection based on quality
    weights = np.array(weights)
    weights = weights ** 2  # Quantum probability amplitudes
    weights = weights / np.sum(weights)
    
    selected_idx = np.random.choice(len(responses), p=weights)
    return responses[selected_idx]

  def process_input(self, user_input: str) -> str:
    """Process user input through quantum-enhanced pipeline"""
    # Generate quantum-enhanced features
    features = self.quantum_enhanced_embedding(user_input)
    qc = self.create_quantum_circuit(features)
    quantum_signature = self.get_quantum_signature(qc)
    
    # Get conversation context from quantum memory
    context_signature = self.quantum_memory.get_context_signature()
    
    # Generate response using quantum superposition
    response = self.generate_superposed_responses(user_input, quantum_signature, context_signature)
    
    # Update quantum memory with this interaction
    self.quantum_memory.update_memory(user_input, response)
    
    return response
  
  def run(self):
    """Main chatbot loop with enhanced quantum reasoning"""
    print("Quantum-Enhanced Chatbot Initialized")
    print("Features: Quantum memory • Superposed responses • Enhanced embeddings")
    print("=" * 65)
    
    try:
      while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in {'exit', 'quit', 'bye', 'goodbye'}:
          print('Quantum: Thank you for the quantum conversation! Goodbye!')
          break
        
        if not user_input:
          continue
        
        try:
          response = self.process_input(user_input)
          print(f"Quantum: {response}")
            
        except Exception as e:
          print(f'Error in quantum processing: {e}')
          print('Please try again...')
    
    except KeyboardInterrupt:
      print('\n\nQuantum: Quantum superposition collapsed. Goodbye!')

if __name__ == '__main__':
  chatbot = QuantumChatbot(max_length = 80)
  chatbot.run()