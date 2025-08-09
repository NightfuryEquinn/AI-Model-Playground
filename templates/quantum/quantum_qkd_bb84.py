import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
import time
from typing import List, Tuple, Dict, Any
import logging

# Configure logging
logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class BB84Protocol:
  def __init__(self, num_bits: int = 64, shots: int = 1024):
    self.num_bits = num_bits
    self.shots = shots
    self.backend = Aer.get_backend('qasm_simulator')
    self.protocol_data = {}

  def generate_random_data(self) -> Tuple[np.ndarray, np.ndarray]:
    """Generate Alice's random bits and bases efficiently"""
    # Use single random call for better performance
    random_data = np.random.randint(2, size = (2, self.num_bits))
    return random_data[0], random_data[1]
  
  def create_quantum_states(self, bits: np.ndarray, bases: np.ndarray) -> List[QuantumCircuit]:
    """Create quantum circuits optimised for batch processing"""
    circuits = []
    state_prep_time = time.time()

    for i, (bit, basis) in enumerate(zip(bits, bases)):
      qc = QuantumCircuit(1, 1)
      qc.name = f"Alice_Qubit_{i}"

      # State Preparation
      if bit == 1:
        qc.x(0) # |0> -> |1>
      
      if basis == 1: # X basis
        qc.h(0) # Apply Hadamard for superposition
      
      circuits.append(qc)
    
    logger.info(f"Quantum state preparation: {time.time() - state_prep_time:.4f}s")
    return circuits
  
  def simulate_eve_attack(
    self, 
    circuits: List[QuantumCircuit],
    eve_present: bool = False
  ) -> Tuple[List[int], np.ndarray]:
    """Simulate Eve's intercept-resend attack with logging"""
    if not eve_present:
      return [], np.array([])
    
    eve_bases = np.random.randint(2, size = self.num_bits)
    eve_results = []
    intercept_time = time.time()

    logger.info("Eve intercepting quantum transmission...")

    for i, qc in enumerate(circuits):
      # Eve's measurement setup
      eve_qc = qc.copy()

      if eve_bases[i] == 1: # X basis measurement
        eve_qc.h(0)

      eve_qc.measure(0, 0)

      # Execute Eve's measurement
      compiled_circuit = transpile(eve_qc, self.backend, optimization_level = 2)
      job = self.backend.run(compiled_circuit, shots = self.shots)
      result = job.result()

      # Get most frequent result (for noise resilience)
      counts = result.get_counts()
      eve_measured = int(max(counts, key = counts.get))
      eve_results.append(eve_measured)

      # Eve prepares new state based on her measurement
      new_qc = QuantumCircuit(1, 1)
      if eve_measured == 1:
        new_qc.x(0)

      if eve_bases[i] == 1:
        new_qc.h(0)

      circuits[i] = new_qc # Replace Alice's state with Eve's prepared state

    logger.info(f"Eve's interception completed: {time.time() - intercept_time:.4f}s")
    return eve_results, eve_bases
  
  def bob_measurement(self, circuits: List[QuantumCircuit]) -> Tuple[List[int], np.ndarray]:
    """Bob's quantum measurements with batch optimisation"""
    bob_bases = np.random.randint(2, size = self.num_bits)
    bob_results = []
    measurement_time = time.time()

    # Batch transpilation for better performance
    compiled_circuits = []
    for i, qc in enumerate(circuits):
      bob_qc = qc.copy()

      if bob_bases[i] == 1: # X basis
        bob_qc.h(0)

      bob_qc.measure(0, 0)
      compiled_circuits.append(transpile(bob_qc, self.backend, optimization_level = 2))

    # Execute all circuits
    for compiled_qc in compiled_circuits:
      job = self.backend.run(compiled_qc, shots = self.shots)
      result = job.result()

      # Majority vote for noise resilience
      counts = result.get_counts()
      measured_value = int(max(counts, key = counts.get))
      bob_results.append(measured_value)

    logger.info(f"Bob's measurements: {time.time() - measurement_time:.4f}s")
    return bob_results, bob_bases
  
  def key_reconciliation(
    self,
    alice_bits: np.ndarray, alice_bases: np.ndarray,
    bob_results: List[int], bob_bases: np.ndarray,
    eve_results: List[int], eve_bases: np.ndarray,
    eve_present: bool
  ) -> Dict[str, Any]:
    """Enhanced key reconciliation with detailed logging"""
    matching_bases = alice_bases == bob_bases
    key_bits = []
    total_errors = 0
    eve_caused_errors = 0

    logger.info("\n" + "="*60)
    logger.info("Key Reconciliation Analysis")
    logger.info("="*60)

    for i in range(self.num_bits):
      alice_bit = alice_bits[i]
      alice_basis = "Z" if alice_bases[i] == 0 else "X"
      bob_result = bob_results[i]
      bob_basis = "Z" if bob_bases[i] == 0 else "X"

      if matching_bases[i]: # Same basis used
        if alice_bit != bob_result:
          total_errors += 1
          error_cause = 'Transmission/Noise'

          # Check if Eve caused this error
          if eve_present and len(eve_results) > 1:
            eve_basis = "Z" if eve_bases[i] == 0 else "X"
            if eve_bases[i] != alice_bases[i]: # Eve used wrong basis
              eve_caused_errors += 1
              error_cause = 'Eve Interference'
          
          logger.info(f"Bit {i:2d}: A({alice_bit},{alice_basis}) → B({bob_result},{bob_basis}) - [{error_cause}]")
        else:
          key_bits.append(bob_result)
          logger.info(f"Bit {i:2d}: A({alice_bit},{alice_basis}) → B({bob_result},{bob_basis}) - [KEY BIT]")
      else:
        logger.info(f"Bit {i:2d}: A({alice_bit}, {alice_basis}) != B(${bob_result}, {bob_basis}) - [BASIS MISMATCH]")

    final_key = "".join(map(str, key_bits))
    error_rate = total_errors / sum(matching_bases) if sum(matching_bases) > 0 else 0

    return {
      'final_key': final_key,
      'key_length': len(key_bits),
      'total_errors': total_errors,
      'eve_caused_errors': eve_caused_errors,
      'error_rate': error_rate,
      'matching_bases_count': sum(matching_bases),
      'security_threshold_exceeded': error_rate > 0.11  # QBER > 11% indicates attack
    }
  
  def run_protocol(self, eve_present: bool = False) -> Dict[str, Any]:
    """Execute complete BB84 protocol with comprehensive logging"""
    start_time = time.time()

    logger.info("STARTING BB84 QUANTUM KEY DISTRIBUTION")
    logger.info(f"Parameters: {self.num_bits} qubits, {self.shots} shots per measurement")
    logger.info(f"Eve present: {'YES' if eve_present else 'NO'}")

    # Alice's preparation
    alice_bits, alice_bases = self.generate_random_data()
    logger.info(f"Alice prepared {self.num_bits} random qubits")

    # Create quantum states
    circuits = self.create_quantum_states(alice_bits, alice_bases)

    # Eve's potential attack
    eve_results, eve_bases = self.simulate_eve_attack(circuits, eve_present)

    # Bob's measurements
    bob_results, bob_bases = self.bob_measurement(circuits)

    # Key reconciliation and analysis
    reconciliation_data = self.key_reconciliation(
      alice_bits, alice_bases, bob_results, bob_bases,
      eve_results, eve_bases, eve_present
    )

    # Final analysis
    total_time = time.time() - start_time

    logger.info("\n" + "-"*60)
    logger.info("PROTOCOL RESULTS")
    logger.info("="*60)
    logger.info(f"Final shared key: {reconciliation_data['final_key']}")
    logger.info(f"Key length: {reconciliation_data['key_length']} bits")
    logger.info(f"Error rate: {reconciliation_data['error_rate']:.3f} ({reconciliation_data['total_errors']} errors)")

    if eve_present:
      logger.info(f"Eve-caused errors: {reconciliation_data['eve_caused_errors']}")

    if reconciliation_data['security_threshold_exceeded']:
      logger.info("SECURITY ALERT: High error rate detected - possible eavesdropping!")
    else:
      logger.info("SECURE: Error rate within acceptable bounds")

    logger.info(f"Protocol execution time: {total_time:.4f}s")

    return {
      **reconciliation_data,
      'alice_bits': alice_bits.tolist(),
      'alice_bases': alice_bases.tolist(),
      'bob_results': bob_results,
      'bob_bases': bob_bases.tolist(),
      'eve_results': eve_results,
      'eve_bases': eve_bases.tolist() if len(eve_bases) > 0 else [],
      'execution_time': total_time,
      'eve_present': eve_present
    }
  
# Example usage and comparison
if __name__ == "__main__":
  # Test without Eve
  print("Testing BB84 without eavesdropper...")
  bb84_secure = BB84Protocol(num_bits = 32, shots = 512)
  secure_result = bb84_secure.run_protocol(eve_present = False)

  print("\n" + "="*80 + "\n")

  # Test with Eve
  print("Testing BB84 with eavesdropper...")
  bb84_attack = BB84Protocol(num_bits = 32, shots = 512)
  attack_result = bb84_attack.run_protocol(eve_present = True)

  print("\n" + "="*80)
  print("PERFORMANCE COMPARISON")
  print("="*80)
  print(f"Secure protocol: {len(secure_result['final_key'])} bit key, {secure_result['error_rate']:.3f} error rate")
  print(f"With attack: {len(attack_result['final_key'])} bit key, {attack_result['error_rate']:.3f} error rate")