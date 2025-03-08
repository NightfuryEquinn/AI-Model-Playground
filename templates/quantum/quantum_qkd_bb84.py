import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer

# Quantum backend
backend = Aer.get_backend('qasm_simulator')

# Number of qubits (bits in key)
num_bits = 12

# Step 1: Alice prepares random qubits
alice_bits = np.random.randint(2, size = num_bits) # Either 0 or 1
alice_bases = np.random.randint(2, size = num_bits) # Chooses a basis (0 = Z. 1 = X)

# Step 2: Create quantum circuits for Alice's qubits
circuits = []
for bit, basis in zip(alice_bits, alice_bases):
  qc = QuantumCircuit(1, 1)
  
  if bit == 1:
    qc.x(0) # Flip to |1> if Alice's bit is 1
  
  if basis == 1:
    qc.h(0) # Apply Hadamard if Alice uses X basis

  circuits.append(qc)

# Simulate eavesdropping
eve_present = True
eve_bases = np.random.randint(2, size = num_bits) if eve_present else None # Eve picks random bases
eve_results = []

if eve_present:
  for i, qc in enumerate(circuits):
    if eve_bases[i] == 1:
      qc.h(0) # Eve applies Hadamard if using X basis
    
    qc.measure(0, 0) # Eve measures qubit, collapsing its state
    
    compiled_circuit = transpile(qc, backend)
    job = backend.run(compiled_circuit)

    result = job.result()
    eve_measured_value = int(list(result.get_counts().keys())[0])
    eve_results.append(eve_measured_value)

# Step 3: Bob randomly chooses measurement bases
bob_bases = np.random.randint(2, size = num_bits)

# Step 4: Run all circuits and simulate transmission
bob_results = []
for i, qc in enumerate(circuits):
  if bob_bases[i] == 1:
    qc.h(0) # Applies Hadamard if measuring in X basis

  qc.measure(0, 0) # Bob measures after Eve (if active)

  compiled_circuit = transpile(qc, backend)
  job = backend.run(compiled_circuit)

  result = job.result()
  bob_results.append(int(list(result.get_counts().keys())[0]))

# Step 5: Key reconciliation - Alica and Bob compare bases
key_bits = []
errors = 0

for i in range(num_bits):
  if alice_bases[i] == bob_bases[i]: # Keep matching bases only
    if eve_present and eve_bases[i] != alice_bases[i]: # If Eve measured in wrong basis
      errors += 1 # Error introduced due to Eve's interference
    else:
      key_bits.append(bob_results[i])

# Step 6: Final secret key
final_key = "".join(map(str, key_bits))
print(f"Shared secret key: {final_key}")
print(f"Errors detected due to Eve: {errors}/{num_bits}")

print(alice_bits)
print(f'[{" ".join(map(str, bob_results))}]')
print(alice_bases)
print(bob_bases)

print(eve_bases)
print(f'[{" ".join(map(str, eve_results))}]')

if errors > 0:
  print("Potential Eavesdropping Detected! Key is NOT secure.")
else:
  print("No interference detected. Key is secure.")