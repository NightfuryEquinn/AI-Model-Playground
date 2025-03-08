import os
from dotenv import load_dotenv
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit_ibm_runtime import QiskitRuntimeService

# Loading IBM Quantum key
load_dotenv()
QISKIT_KEY = os.getenv('QISKIT_KEY')

# Load account
provider = QiskitRuntimeService(channel = 'ibm_quantum', token = QISKIT_KEY)

# Create Quantum circuit acting on Q register
circuit = QuantumCircuit(2, 2)

# Add a H gate on qubit 0
circuit.h(0)

# Add a CX (CNOT) gate on control qubit 0 and target qubit 1
circuit.cx(0, 1)

# Map the quantum measurement to the classical bits
circuit.measure([0, 1], [0, 1])

# Use Aer's qasm_simulator
backend = Aer.get_backend('qasm_simulator')

# Transpile the circuit for the simulator
compiled_circuit = transpile(circuit, backend)

# Execute the circuit
job = backend.run(compiled_circuit)

# Grab results from the job
result = job.result()

# Return counts
counts = result.get_counts()
print(f"Total count for 00 and 11 are: ", counts)

# Draw circuit
print(circuit.draw())