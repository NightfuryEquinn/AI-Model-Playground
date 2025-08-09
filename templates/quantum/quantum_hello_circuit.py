from qiskit_ibm_runtime.sampler import SamplerV2
import os
from dotenv import load_dotenv
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.converters import circuit_to_dag

# Loading IBM Quantum key
load_dotenv()
QISKIT_KEY = os.getenv('QISKIT_KEY')

# Load account
# Need to create at IBM Quantum Platform website
provider = QiskitRuntimeService(channel = 'ibm_quantum_platform', token = QISKIT_KEY)

# Create Quantum circuit
# A quantum circuit with 2 qubits and 2 classical bits
# h(qubit) applies the Hadamard gate, putting qubit into superposition state
# cx(control, target) applies a CNOT gate, entangling the two qubits
# measure(q, c) measure qubits into classical bits
# Use case: Creates a simple Bell state (entangle qubits) and measures the results
circuit = QuantumCircuit(2, 2)
circuit.h(0)
circuit.h(1)
circuit.cx(0, 1)
circuit.measure([0, 1], [0, 1])

# Local simulator
# transpile (circuit, backend) converts the circuit into instructions compatible with backend
# Use case: Allows fast, cost-free testing of circuit locally before running on real quantum hardware
backend_sim = Aer.get_backend('qasm_simulator')
compiled_circuit = transpile(circuit, backend_sim)
job = backend_sim.run(compiled_circuit)
# Returns the histogram of measurement outcomes
result = job.result()
counts = result.get_counts()
print(f"Local simulator counts: {counts}")

# IBM Real backend
# transpile(..., optimization_level) optimise the circuit for device to reduce gate errors and execution time
# Gate error refers to the probability that quantum gate (H, CX etc.) produce wrong result when applied to real quantum hardware
# Real quantum device are noisy due to imperfect control pulses, qubit decoherence, crosstalk and hardware limitations
# Each gate has a fidelity (accuracy) < 100%
# Use case: Runs the optimised circuit on real quantum computer to get real-world measurement results
backend = provider.backend("ibm_brisbane")
transpiled_circuit = transpile(circuit, backend = backend, optimization_level = 3)
# Find gates in transpiled circuit
dag = circuit_to_dag(transpiled_circuit)
used_gates = set([node.name for node in dag.op_nodes()])
print("Used gates:", used_gates)
# Get backend properties
properties = backend.properties()
print("\nGate Error Rates (only gates used in circuit):")
# Store total fidelity (probability of NO error)
total_fidelity = 1.0
# Loop through gate errors
# Important for circuit reliability -> More gates, more accumulated errors
# Backend choice -> Different devices, different error rates
# Optimisation -> Transpilation can reduce high-error gates like replace multiple CNOTs with fewer
for gate in properties.gates:
  if gate.gate in used_gates:
    error_info = [p for p in gate.parameters if p.name == "gate_error"]
    if error_info:
      error_rate = error_info[0].value
      print(f"Gate: {gate.gate} on qubits {gate.qubits} → Error rate: {error_rate:.5%}")
      total_fidelity *= (1 - error_rate)
# Readout error calculation
# The probability that quantum hardware will report the wrong classical measurement value
# Even if quantum state in qubit was correct before measurement
print("\nReadout Error Rates:")
for qubit, qubit_props in enumerate(properties.qubits):
  readout_info = [p for p in qubit_props if p.name == "readout_error"]
  if readout_info:
    readout_error = readout_info[0].value
    print(f"Qubit {qubit} readout error: {readout_error:.5%}")
    total_fidelity *= (1 - readout_error)

# Execute transpiled circuit 1024 times
sampler = SamplerV2(backend)
job = sampler.run([transpiled_circuit], shots = 1024)
result = job.result()
# Access measurement data for SamplerV2
# The classical measurement register data
measurement_data = result[0].data.c
counts = measurement_data.get_counts()
print("IBM backend counts:", counts)

# Estimated total error
total_error = 1 - total_fidelity
print(f"\nEstimated total circuit error probability: {total_error:.3%}")

# Draw quantum circuit
print("\nQuantum Circuit:")
print(circuit.draw(output = 'text'))