import os
from dotenv import load_dotenv
from qiskit import QuantumCircuit
from qiskit.primitives import BackendSampler
from qiskit_ibm_provider import IBMProvider as IBMQ
from qiskit_aer import Aer

# Loading IBM Quantum key
load_dotenv()
QISKIT_KEY = os.getenv('QISKIT_KEY')

# Access IBM Quantum account
if not IBMQ.saved_accounts():
  IBMQ.save_account(QISKIT_KEY)

# Load account
provider = IBMQ()

# Create Quantum circuit acting on Q register
circuit = QuantumCircuit(2, 2)

# Add a H gate on qubit 0
circuit.h(0)

# Add a CX (CNOT) gate on control qubit 0 and target qubit 1
circuit.cx(0, 1)

# Map the quantum measurement to the classical bits
circuit.measure([0, 1], [0, 1])

# Use Aer's qasm_simulator
simulator = Aer.get_backend('qasm_simulator')

# Execute the cirucit on the qasm simulator
sampler = BackendSampler(simulator)
job = sampler.run([circuit])

# Grab results from the job
result = job.result()

# Return counts
counts = result.quasi_dists[0]
print(f"Total count for 00 and 11 are: ", counts)

# Draw circuit
print(circuit.draw())