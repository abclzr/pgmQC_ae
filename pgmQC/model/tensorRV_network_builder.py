import numpy as np
import pdb
import copy
from typing import List, Tuple
import cotengra as ctg
import quimb.tensor as qtn
from tqdm import tqdm
import time

from qiskit import dagcircuit, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import CircuitInstruction
from qiskit.converters import circuit_to_dag, dag_to_circuit
from pgmpy.models import MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor
from qiskit.quantum_info import Pauli, SparsePauliOp, DensityMatrix, random_clifford
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit_ibm_runtime import SamplerV2, Batch
from pgmQC.utils.setting import I, X, Y, Z
from qiskit.circuit.parameter import Parameter

class TensorIndex:
    def __init__(self, dim, index):
        if isinstance(index, list):
            self.decomposed_index = index
            self.merged_index = self.get_merged_index()
        else:
            # Otherwise, decompose the index one dimension by one dimension
            self.merged_index = index
            self.decomposed_index = []
            for i in range(dim):
                self.decomposed_index.append(index % 4)
                index = index // 4
            self.decomposed_index = list(reversed(self.decomposed_index))
    
    def get_merged_index(self) -> int:
        merged_index = 0
        for _ in self.decomposed_index:
            merged_index = merged_index * 4 + _
        return merged_index
    
    def __add__(self, other):
        if not isinstance(other, TensorIndex):
            raise ValueError("Operand must be an instance of TensorIndex")
        
        new_decomposed_index = self.decomposed_index + other.decomposed_index
        return TensorIndex(len(new_decomposed_index), new_decomposed_index)
    
    def get_subindex(self, permutation: List[int]) -> List[int]:
        subindex = []
        for _ in permutation:
            subindex.append(self.decomposed_index[_])
        return subindex
    
    def to_paulistring(self) -> str:
        return ''.join(['I', 'X', 'Y', 'Z'][i] for i in self.decomposed_index)
    
    def __eq__(self, other):
        return self.merged_index == other.merged_index

    def __hash__(self):
        return self.merged_index

    def __str__(self):
        return 'TensorIndex' + str(self.decomposed_index)
    
    def __repr__(self):
        return 'TensorIndex' + str(self.decomposed_index)
    
class TensorRV:
    def __init__(self, mean, cov, variables):
        if isinstance(mean, list):
            self.mean = mean
        else:
            assert isinstance(mean, np.ndarray)
            self.mean = []
            for key, value in enumerate(mean.reshape(-1)):
                if value != 0:
                    index = TensorIndex(len(variables), key)
                    self.mean.append((index, value))
        if isinstance(cov, list):
            self.cov = cov
        else:
            assert isinstance(cov, np.ndarray)
            self.cov = []
            for key, value in enumerate(cov.reshape(-1)):
                if value != 0:
                    index = TensorIndex(len(variables) * 2, key)
                    self.cov.append((index, value))
        self.variables = variables
    
    def mean_of_trace_as_scalar(self, num_qubits):
        assert len(self.mean) <= 1
        if len(self.mean) == 0:
            return 0
        return self.mean[0][1] * (2 ** num_qubits)
    
    def non_zero_means_of_trace(self, num_qubits):
        ret = []
        for index, value in self.mean:
            ret.append((index.to_paulistring(), value * (2 ** num_qubits)))
        return ret
    
    def variance_of_trace_as_scalar(self, num_qubits):
        assert len(self.cov) <= 1
        if len(self.cov) == 0:
            return 0
        return self.cov[0][1] * (4 ** num_qubits)
    
    def paulistrings_and_variances_of_trace(self, num_qubits):
        ret = []
        for index, value in self.cov:
            n = len(index.decomposed_index)
            if np.all(index.decomposed_index[:n//2] == index.decomposed_index[n//2:]):
                ps = ''
                for i in index.decomposed_index[:n//2]:
                    ps = ps + ['I', 'X', 'Y', 'Z'][i]
                ret.append((ps, value * (4 ** num_qubits)))
            else:
                raise ValueError("The index is not a variance index")
        return ret
    
    def __mul__(self, scalar):
        if not isinstance(scalar, (int, float)):
            raise ValueError("Operand must be a scalar")
        
        new_mean = [(index, value * scalar) for index, value in self.mean]
        new_cov = [(index, value * scalar) for index, value in self.cov]
        
        return TensorRV(new_mean, new_cov, self.variables)
    
    def __truediv__(self, scalar):
        if not isinstance(scalar, (int, float)):
            raise ValueError("Operand must be a scalar")
        
        new_mean = [(index, value / scalar) for index, value in self.mean]
        new_cov = [(index, value / scalar) for index, value in self.cov]
        
        return TensorRV(new_mean, new_cov, self.variables)
    
    def __sub__(self, other):
        if not isinstance(other, TensorRV):
            raise ValueError("Operand must be an instance of TensorRV")
        
        if self.variables != other.variables:
            raise ValueError("Both TensorRV instances must have the same variables")
        
        new_mean = {}
        for index, value in self.mean:
            new_mean[index] = value
        for index, value in other.mean:
            if index in new_mean:
                new_mean[index] -= value
            else:
                new_mean[index] = -value
        
        new_cov = {}
        for index, value in self.cov:
            new_cov[index] = value
        for index, value in other.cov:
            if index in new_cov:
                new_cov[index] -= value
            else:
                new_cov[index] = -value
        
        E_C = [(index, value) for index, value in new_mean.items() if np.abs(value) > 1e-16]
        Cov_C = [(index, value) for index, value in new_cov.items() if np.abs(value) > 1e-16]
        
        return TensorRV(E_C, Cov_C, self.variables)
    
    def __add__(self, other):
        if not isinstance(other, TensorRV):
            raise ValueError("Operand must be an instance of TensorRV")
        
        if self.variables != other.variables:
            raise ValueError("Both TensorRV instances must have the same variables")
        
        new_mean = {}
        for index, value in self.mean:
            new_mean[index] = value
        for index, value in other.mean:
            if index in new_mean:
                new_mean[index] += value
            else:
                new_mean[index] = value
        
        new_cov = {}
        for index, value in self.cov:
            new_cov[index] = value
        for index, value in other.cov:
            if index in new_cov:
                new_cov[index] += value
            else:
                new_cov[index] = value
        
        E_C = [(index, value) for index, value in new_mean.items() if np.abs(value) > 1e-16]
        Cov_C = [(index, value) for index, value in new_cov.items() if np.abs(value) > 1e-16]
        
        return TensorRV(E_C, Cov_C, self.variables)
    
    def add_cov_index_value(self, index, value):
        if not isinstance(index, TensorIndex):
            raise ValueError("Index must be an instance of TensorIndex")
        if not isinstance(value, (int, float)):
            raise ValueError("Value must be a scalar")
        
        for i in range(len(self.cov)):
            if self.cov[i][0] == index:
                self.cov[i] = (index, self.cov[i][1] + value)
                return
        self.cov.append((index, value))
    
    def __matmul__(self, other):
        if not isinstance(other, TensorRV):
            raise ValueError("Operand must be an instance of TensorRV")
        
        E_A = self.mean
        Cov_A = self.cov
        E_B = other.mean
        Cov_B = other.cov
        
        common_vars = []
        for var in other.variables:
            if var in self.variables:
                common_vars.append(var)
        uncommon_vars_A = []
        for var in self.variables:
            if var not in other.variables:
                uncommon_vars_A.append(var)
        uncommon_vars_B = []
        for var in other.variables:
            if var not in self.variables:
                uncommon_vars_B.append(var)
        
        new_vars = uncommon_vars_A + uncommon_vars_B
        
        common_permutation_A = []
        for var in common_vars:
            common_permutation_A.append(self.variables.index(var))
        uncommon_permutation_A = []
        for var in uncommon_vars_A:
            uncommon_permutation_A.append(self.variables.index(var))
        common_permutation_B = []
        for var in common_vars:
            common_permutation_B.append(other.variables.index(var))
        uncommon_permutation_B = []
        for var in uncommon_vars_B:
            uncommon_permutation_B.append(other.variables.index(var))
        
        new_mean = {}
        for index1, value1 in E_A:
            index_uncommon_A = index1.get_subindex(uncommon_permutation_A)
            index_common_A = index1.get_subindex(common_permutation_A)
            for index2, value2 in E_B:
                index_uncommon_B = index2.get_subindex(uncommon_permutation_B)
                index_common_B = index2.get_subindex(common_permutation_B)
                if index_common_A == index_common_B:
                    new_index = TensorIndex(len(new_vars), index_uncommon_A + index_uncommon_B)
                    if new_index in new_mean.keys():
                        new_mean[new_index] += value1 * value2
                    else:
                        new_mean[new_index] = value1 * value2
        
        cov_uncommon_permutation_A = uncommon_permutation_A + [i + len(self.variables) for i in uncommon_permutation_A]
        cov_uncommon_permutation_B = uncommon_permutation_B + [i + len(other.variables) for i in uncommon_permutation_B]
        cov_common_permutation_A = common_permutation_A + [i + len(self.variables) for i in common_permutation_A]
        cov_common_permutation_B = common_permutation_B + [i + len(other.variables) for i in common_permutation_B]
        new_cov = {}
        # Cov(C_ij, C_qs) += Σ_k Σ_r { Cov(A_ik, A_qr) * E(B_kj) * E(B_rs) }
        for index1, value1 in Cov_A:
            index_i_q = index1.get_subindex(cov_uncommon_permutation_A)
            index_i = index_i_q[:len(uncommon_permutation_A)]
            index_q = index_i_q[len(uncommon_permutation_A):]
            index_k_r = index1.get_subindex(cov_common_permutation_A)
            index_k = index_k_r[:len(common_permutation_A)]
            index_r = index_k_r[len(common_permutation_A):]
            for index2, value2 in E_B:
                index_j = index2.get_subindex(uncommon_permutation_B)
                index_k2 = index2.get_subindex(common_permutation_B)
                if index_k == index_k2:
                    for index3, value3 in E_B:
                        index_s = index3.get_subindex(uncommon_permutation_B)
                        index_r2 = index3.get_subindex(common_permutation_B)
                        if index_r == index_r2:
                            new_index = TensorIndex(len(new_vars) * 2, index_i + index_j + index_q + index_s)
                            if new_index in new_cov.keys():
                                new_cov[new_index] += value1 * value2 * value3
                            else:
                                new_cov[new_index] = value1 * value2 * value3
        # Cov(C_ij, C_qs) += Σ_k Σ_r { E(A_ik) * E(A_qr) * Cov(B_kj, B_rs) }
        for index1, value1 in Cov_B:
            index_k_r = index1.get_subindex(cov_common_permutation_B)
            index_k = index_k_r[:len(common_permutation_B)]
            index_r = index_k_r[len(common_permutation_B):]
            index_j_s = index1.get_subindex(cov_uncommon_permutation_B)
            index_j = index_j_s[:len(uncommon_permutation_B)]
            index_s = index_j_s[len(uncommon_permutation_B):]
            for index2, value2 in E_A:
                index_i = index2.get_subindex(uncommon_permutation_A)
                index_k2 = index2.get_subindex(common_permutation_A)
                if index_k == index_k2:
                    for index3, value3 in E_A:
                        index_q = index3.get_subindex(uncommon_permutation_A)
                        index_r2 = index3.get_subindex(common_permutation_A)
                        if index_r == index_r2:
                            new_index = TensorIndex(len(new_vars) * 2, index_i + index_j + index_q + index_s)
                            if new_index in new_cov.keys():
                                new_cov[new_index] += value1 * value2 * value3
                            else:
                                new_cov[new_index] = value1 * value2 * value3
        # Cov(C_ij, C_qs) += Σ_k Σ_r { Cov(A_ik, A_qr) * Cov(B_kj, B_rs) }
        for index1, value1 in Cov_A:
            index_i_q = index1.get_subindex(cov_uncommon_permutation_A)
            index_i = index_i_q[:len(uncommon_permutation_A)]
            index_q = index_i_q[len(uncommon_permutation_A):]
            index_k_r = index1.get_subindex(cov_common_permutation_A)
            index_k = index_k_r[:len(common_permutation_A)]
            index_r = index_k_r[len(common_permutation_A):]
            for index2, value2 in Cov_B:
                index_k_r = index2.get_subindex(cov_common_permutation_B)
                index_k2 = index_k_r[:len(common_permutation_B)]
                index_r2 = index_k_r[len(common_permutation_B):]
                index_j_s = index2.get_subindex(cov_uncommon_permutation_B)
                index_j = index_j_s[:len(uncommon_permutation_B)]
                index_s = index_j_s[len(uncommon_permutation_B):]
                if index_k == index_k2 and index_r == index_r2:
                    new_index = TensorIndex(len(new_vars) * 2, index_i + index_j + index_q + index_s)
                    if new_index in new_cov.keys():
                        new_cov[new_index] += value1 * value2
                    else:
                        new_cov[new_index] = value1 * value2
        E_C = []
        for key, value in new_mean.items():
            if np.abs(value) > 1e-16:
                E_C.append((key, value))
        Cov_C = []
        for key, value in new_cov.items():
            if np.abs(value) > 1e-16:
                Cov_C.append((key, value))
        return TensorRV(E_C, Cov_C, new_vars)

class TensorBoolean:
    def __init__(self, models, variables):
        if isinstance(models, list):
            self.models = models
        else:
            assert isinstance(models, np.ndarray)
            self.models = []
            for key, value in enumerate(models.reshape(-1)):
                if value != 0:
                    index = TensorIndex(len(variables), key)
                    self.models.append(index)
        self.variables = variables
    
    def paulistrings(self):
        ret = []
        for index in self.models:
            ps = ''
            for i in index.decomposed_index:
                ps = ps + ['I', 'X', 'Y', 'Z'][i]
            ret.append(ps)
        return ret
    
    def __matmul__(self, other):
        if not isinstance(other, TensorBoolean):
            raise ValueError("Operand must be an instance of TensorBoolean")
        
        models_A = self.models
        models_B = other.models
        
        common_vars = []
        for var in other.variables:
            if var in self.variables:
                common_vars.append(var)
        uncommon_vars_A = []
        for var in self.variables:
            if var not in other.variables:
                uncommon_vars_A.append(var)
        uncommon_vars_B = []
        for var in other.variables:
            if var not in self.variables:
                uncommon_vars_B.append(var)
        
        new_vars = uncommon_vars_A + uncommon_vars_B
        
        common_permutation_A = []
        for var in common_vars:
            common_permutation_A.append(self.variables.index(var))
        uncommon_permutation_A = []
        for var in uncommon_vars_A:
            uncommon_permutation_A.append(self.variables.index(var))
        common_permutation_B = []
        for var in common_vars:
            common_permutation_B.append(other.variables.index(var))
        uncommon_permutation_B = []
        for var in uncommon_vars_B:
            uncommon_permutation_B.append(other.variables.index(var))
        
        contracted_models = set()
        for index1 in models_A:
            index_uncommon_A = index1.get_subindex(uncommon_permutation_A)
            index_common_A = index1.get_subindex(common_permutation_A)
            for index2 in models_B:
                index_uncommon_B = index2.get_subindex(uncommon_permutation_B)
                index_common_B = index2.get_subindex(common_permutation_B)
                if index_common_A == index_common_B:
                    new_index = TensorIndex(len(new_vars), index_uncommon_A + index_uncommon_B)
                    contracted_models.add(new_index)
        
        E_C = list(contracted_models)
        return TensorBoolean(E_C, new_vars)


class TensorRVNetworkBuilder:
    def __init__(self, dag: DAGCircuit, prefix='v'):
        self.dag = dag
        self.dag.draw(filename="dag.png")
        self.pauli_index_lookup = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}
        self.nodes = list(self.dag.nodes())
        self.vars = list(self.dag.edges())
        self.var_name_lookup = {}
        for i, var in enumerate(self.vars):
            self.var_name_lookup[var] = f'{prefix}_{i}'

    # get the var name of an edge
    def get_var_name(self, edge):
        return self.var_name_lookup[edge]
    
    """
        Rz(θ)•I•Rz^dag(θ) = I
        Rz(θ)•X•Rz^dag(θ) = cos(θ)X + sin(θ)Y
        Rz(θ)•Y•Rz^dag(θ) = -sin(θ)X + cos(θ)Y
        Rz(θ)•Z•Rz^dag(θ) = Z
    """
    @staticmethod
    def get_RZ():
        mean = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
        cov = np.zeros((4, 4, 4, 4))
        cov[1, 1, 1, 1] = .5
        cov[1, 2, 1, 2] = .5
        cov[2, 1, 2, 1] = .5
        cov[2, 2, 2, 2] = .5
        cov[1, 1, 2, 2] = .5
        cov[2, 2, 1, 1] = .5
        cov[1, 2, 2, 1] = -.5
        cov[2, 1, 1, 2] = -.5
        return mean, cov

    """
        Rx(θ)•I•Rx^dag(θ) = I
        Rx(θ)•X•Rx^dag(θ) = X
        Rx(θ)•Y•Rx^dag(θ) = cos(θ)Y + sin(θ)Z
        Rx(θ)•Z•Rx^dag(θ) = -sin(θ)Y + cos(θ)Z
    """
    @staticmethod
    def get_RX():
        mean = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        cov = np.zeros((4, 4, 4, 4))
        cov[2, 2, 2, 2] = .5
        cov[2, 3, 2, 3] = .5
        cov[3, 2, 3, 2] = .5
        cov[3, 3, 3, 3] = .5
        cov[2, 2, 3, 3] = .5
        cov[3, 3, 2, 2] = .5
        cov[2, 3, 3, 2] = -.5
        cov[3, 2, 2, 3] = -.5
        return mean, cov

    """
        Ry(θ)•I•Ry^dag(θ) = I
        Ry(θ)•X•Ry^dag(θ) = cos(θ)X - sin(θ)Z
        Ry(θ)•Y•Ry^dag(θ) = Y
        Ry(θ)•Z•Ry^dag(θ) = sin(θ)X + cos(θ)Z
    """
    @staticmethod
    def get_RY():
        mean = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
        cov = np.zeros((4, 4, 4, 4))
        cov[1, 1, 1, 1] = .5
        cov[1, 3, 1, 3] = .5
        cov[3, 1, 3, 1] = .5
        cov[3, 3, 3, 3] = .5
        cov[1, 1, 3, 3] = .5
        cov[3, 3, 1, 1] = .5
        cov[1, 3, 3, 1] = -.5
        cov[3, 1, 1, 3] = -.5
        return mean, cov

    """
        Rz(θ)•I•Rz^dag(θ) = I
        Rz(θ)•X•Rz^dag(θ) = cos(θ)X + sin(θ)Y
        Rz(θ)•Y•Rz^dag(θ) = -sin(θ)X + cos(θ)Y
        Rz(θ)•Z•Rz^dag(θ) = Z
    """
    @staticmethod
    def get_differentiated_RZ():
        mean = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        cov = np.zeros((4, 4, 4, 4))
        cov[1, 1, 1, 1] = .5
        cov[1, 2, 1, 2] = .5
        cov[2, 1, 2, 1] = .5
        cov[2, 2, 2, 2] = .5
        cov[1, 1, 2, 2] = .5
        cov[2, 2, 1, 1] = .5
        cov[1, 2, 2, 1] = -.5
        cov[2, 1, 1, 2] = -.5
        return mean, cov

    """
        Rx(θ)•I•Rx^dag(θ) = I
        Rx(θ)•X•Rx^dag(θ) = X
        Rx(θ)•Y•Rx^dag(θ) = cos(θ)Y + sin(θ)Z
        Rx(θ)•Z•Rx^dag(θ) = -sin(θ)Y + cos(θ)Z
    """
    @staticmethod
    def get_differentiated_RX():
        mean = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        cov = np.zeros((4, 4, 4, 4))
        cov[2, 2, 2, 2] = .5
        cov[2, 3, 2, 3] = .5
        cov[3, 2, 3, 2] = .5
        cov[3, 3, 3, 3] = .5
        cov[2, 2, 3, 3] = .5
        cov[3, 3, 2, 2] = .5
        cov[2, 3, 3, 2] = -.5
        cov[3, 2, 2, 3] = -.5
        return mean, cov

    """
        Ry(θ)•I•Ry^dag(θ) = I
        Ry(θ)•X•Ry^dag(θ) = cos(θ)X - sin(θ)Z
        Ry(θ)•Y•Ry^dag(θ) = Y
        Ry(θ)•Z•Ry^dag(θ) = sin(θ)X + cos(θ)Z
    """
    @staticmethod
    def get_differentiated_RY():
        mean = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        cov = np.zeros((4, 4, 4, 4))
        cov[1, 1, 1, 1] = .5
        cov[1, 3, 1, 3] = .5
        cov[3, 1, 3, 1] = .5
        cov[3, 3, 3, 3] = .5
        cov[1, 1, 3, 3] = .5
        cov[3, 3, 1, 1] = .5
        cov[1, 3, 3, 1] = -.5
        cov[3, 1, 1, 3] = -.5
        return mean, cov

    '''
        For a depolarizing channel-like random Pauli rotation gate, name RD:
        The tensor of RD = 1/3 RX + 1/3 RY + 1/3 RZ
        RD(θ)•I•RD^dag(θ) = I
        RD(θ)•X•RD^dag(θ) = (2/3cos(θ)+1/3) X + 1/3sin(θ) Y - 1/3sin(θ) Z
        RD(θ)•Y•RD^dag(θ) = -1/3sin(θ) X + (2/3cos(θ)+1/3) Y + 1/3sin(θ) Z
        RD(θ)•Z•RD^dag(θ) = 1/3sin(θ) X - 1/3sin(θ) Y + (2/3cos(θ)+1/3) Z
    '''
    @staticmethod
    def get_RD():
        mean = np.array([[1, 0, 0, 0], [0, 1./3., 0, 0], [0, 0, 1./3., 0], [0, 0, 0, 1./3.]])
        cov = np.zeros((4, 4, 4, 4))
        cov[1, 1, 1, 1] = 2./9.
        cov[2, 2, 2, 2] = 2./9.
        cov[3, 3, 3, 3] = 2./9.
        
        cov[1, 1, 2, 2] = 2./9.
        cov[2, 2, 1, 1] = 2./9.
        cov[1, 1, 3, 3] = 2./9.
        cov[3, 3, 1, 1] = 2./9.
        cov[2, 2, 3, 3] = 2./9.
        cov[3, 3, 2, 2] = 2./9.
        
        axis = [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]
        sign = [1, -1, -1, 1, 1, -1]
        for a, sign_a in zip(axis, sign):
            for b, sign_b in zip(axis, sign):
                cov[a[0], a[1], b[0], b[1]] = sign_a * sign_b / 18.
        return mean, cov
    
    @staticmethod
    def merge_channels(*tensorRV_args):
        tensorRV_list = list(tensorRV_args)
        n_channels = len(tensorRV_list)
        differentiated_tensorRV_list = []
        for i in range(len(tensorRV_list)-1):
            differentiated_tensorRV_list.append(tensorRV_list[i] - tensorRV_list[i+1])
        differentiated_tensorRV_list.append(tensorRV_list[-1] - tensorRV_list[0])
        new_tensorRV = differentiated_tensorRV_list[0]
        for i in range(1, n_channels):
            new_tensorRV = new_tensorRV + differentiated_tensorRV_list[i]
        new_tensorRV = new_tensorRV / n_channels
        for differentiated_tensorRV in differentiated_tensorRV_list:
            for index1, value1 in differentiated_tensorRV.mean:
                for index2, value2 in differentiated_tensorRV.mean:
                    new_tensorRV.add_cov_index_value(index1 + index2, value1 * value2 / n_channels / n_channels)
        return new_tensorRV
    
    # build the markov network
    # enable: a dictionary with the key as the node and the value as a boolean
    #         if the value is True, the node shall be enabled.
    #         if the value is False, the node shall be disabled.
    #         if the value is None, the node shall be enabled.
    #         dagcircuit.DAGInNode shall always be enabled.var_name(edge)
    # dagcircuit.DAGOutNode shall never be enabled.
    def build(self, observable=None):
        nodes = self.nodes
        vars = self.vars
        var_name_lookup = self.var_name_lookup
        pauli_index_lookup = self.pauli_index_lookup
        
        markov_net = MarkovNetwork()
        factors = set()
        self.true_false_network = MarkovNetwork()
        
        def connected(opnode):
            connected_vars = []
            for var in vars:
                if var[0] == opnode or var[1] == opnode:
                    connected_vars.append(var)
            return connected_vars
        
        tensorRV_list = []
        tensorBoolean_list = []
        output_var_dict = {}
        for opnode in nodes:
            if isinstance(opnode, dagcircuit.DAGInNode):
                connected_vars = connected(opnode)
                if var_name_lookup[connected_vars[0]] == 'this_edge_should_not_exists':
                    continue
                factor = DiscreteFactor([var_name_lookup[connected_vars[0]]],
                        cardinality=[4],
                        values=[.5, 0, 0, .5],
                        state_names={var_name_lookup[var] : ['I', 'X', 'Y', 'Z'] for var in connected_vars}
                )
                factors.add(factor)
                tensorRV_list.append(TensorRV(np.array([.5, 0, 0, .5]), np.zeros([4, 4]), [var_name_lookup[connected_vars[0]]]))
                tensorBoolean_list.append(TensorBoolean(np.array([.5, 0, 0, .5]), [var_name_lookup[connected_vars[0]]]))
            elif isinstance(opnode, dagcircuit.DAGOpNode):
                if opnode.op.name == 'qpd_1q':
                    continue
                # sort the connected variables based on the opnode.qargs
                connected_vars = connected(opnode)
                connected_vars = [(var, var[1] == opnode, opnode.qargs.index(var[2])) for var in connected_vars]
                connected_vars = sorted(connected_vars, key = lambda x: (not x[1], x[2]))
                connected_vars = [var[0] for var in connected_vars]
                
                # calculate the values in the factors
                
                # initialize the input variables state
                paulis = [ Pauli('I'), Pauli('X'), Pauli('Y'), Pauli('Z') ]
                strings = paulis

                n = opnode.op.num_qubits
                while n > 1:
                    new_strings = []
                    for a in strings:
                        for b in paulis:
                            new_strings.append(a.expand(b))
                    strings = new_strings
                    n = n-1
                
                has_parameter = False
                for param in opnode.op.params:
                    if isinstance(param, Parameter):
                        has_parameter = True
                        break
                if has_parameter:
                    assert opnode.op.name in ['rx', 'ry', 'rz']
                    assert len(opnode.op.params) == 1
                    if opnode.op.name == 'rx':
                        mean, cov = self.get_RX()
                        values = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]])
                    elif opnode.op.name == 'ry':
                        mean, cov = self.get_RY()
                        values = np.array([[1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 1, 0, 1]])                        
                    elif opnode.op.name == 'rz':
                        mean, cov = self.get_RZ()
                        values = np.array([[1, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 1]])
                    factor = DiscreteFactor([var_name_lookup[var] for var in connected_vars],
                            cardinality=[4 for var in connected_vars],
                            values=values,
                            state_names={var_name_lookup[var] : ['I', 'X', 'Y', 'Z'] for var in connected_vars}
                    )
                    factors.add(factor)
                    tensorRV = TensorRV(mean, cov, [var_name_lookup[var] for var in connected_vars])
                    tensorBoolean = TensorBoolean(values, [var_name_lookup[var] for var in connected_vars])
                else:
                    # for every input variables state, get the output variables state
                    values = []
                    for string in strings:
                        sparse = SparsePauliOp.from_operator(DensityMatrix(string).evolve(opnode.op))
                        out_state_values = np.zeros(4 ** opnode.op.num_qubits)
                        for label, coeff in sparse.label_iter(): # type: ignore
                            index = 0
                            # neccecary to reverse the label, because qiskit's SparsePauliOp('IX') will be 'X' on qubit 0 and 'I' on qubit 1
                            for pauli in reversed(label):
                                index = index * 4 + pauli_index_lookup[pauli]
                            out_state_values[index] = coeff
                        
                        values.append(out_state_values)

                    # create the factor
                    values = np.concatenate(values, axis=0).real
                    factor = DiscreteFactor([var_name_lookup[var] for var in connected_vars],
                            cardinality=[4 for var in connected_vars],
                            values=values.real,
                            state_names={var_name_lookup[var] : ['I', 'X', 'Y', 'Z'] for var in connected_vars}
                    )
                    factors.add(factor)
                    tensorRV = TensorRV(values, np.zeros([4] * (2 * opnode.op.num_qubits)), [var_name_lookup[var] for var in connected_vars])
                    tensorBoolean = TensorBoolean(values, [var_name_lookup[var] for var in connected_vars])
                tensorRV_list.append(tensorRV)
                tensorBoolean_list.append(tensorBoolean)
            elif isinstance(opnode, dagcircuit.DAGOutNode):
                connected_vars = connected(opnode)
                if var_name_lookup[connected_vars[0]] == 'this_edge_should_not_exists':
                    continue
                output_var_dict[connected_vars[0][1].wire._index] = var_name_lookup[connected_vars[0]]
                # factor = DiscreteFactor([var_name_lookup[connected_vars[0]]],
                #     cardinality=[4],
                #     values=[0, 0, 0, 1.],
                #     state_names={var_name_lookup[var] : ['I', 'X', 'Y', 'Z'] for var in connected_vars}
                # )
                # factors.add(factor)
            else:
                raise TypeError(opnode)

        for var in vars:
            if var_name_lookup[var] == 'this_edge_should_not_exists':
                continue
            markov_net.add_node(var_name_lookup[var])
            self.true_false_network.add_node(var_name_lookup[var])

        for factor in factors:
            markov_net.add_factors(factor)
            self.true_false_network.add_factors(
                DiscreteFactor(factor.variables,
                    factor.cardinality,
                    np.where(factor.values != 0, 1, 0).real,
                    factor.state_names
                )
            )
            edges = [(factor.variables[i], factor.variables[j])
                    for i in range(len(factor.variables))
                    for j in range(i + 1, len(factor.variables))]
            markov_net.add_edges_from(ebunch=edges)
            self.true_false_network.add_edges_from(ebunch=edges)
        
        nodes_count = {var_name: 2 for var_name in var_name_lookup.values() if var_name != 'this_edge_should_not_exists'}

        for tensorRV in tensorRV_list:
            for var_name in tensorRV.variables:
                nodes_count[var_name] -= 1
        
        uncontracted_vars = []
        for node, count in nodes_count.items():
            if count == 1:
                uncontracted_vars.append(node)
            else:
                assert count == 0
        
        assert set(output_var_dict.values()) == set(uncontracted_vars)
        self.tensorRV_list = tensorRV_list
        self.tensorBoolean_list = tensorBoolean_list
        if not 0 in output_var_dict.keys():
            pdb.set_trace()
        self.uncontracted_vars = [output_var_dict[i] for i in reversed(range(len(output_var_dict)))]
        return self.true_false_network, uncontracted_vars, tensorRV_list

    def _contract_tensors(self, tensorRV_list=None, uncontracted_vars=None):
        """
        Calculate the path of contracting a list of tensors, and then contract them.
        
        Args:
            tensor_list: List of tensors to contract.
            uncontracted_vars: List of uncontracted variables.
        
        Returns:
            A tensor representing the result of the contraction.
        """
        if tensorRV_list is None:
            tensorRV_list = self.tensorRV_list
        if uncontracted_vars is None:
            uncontracted_vars = self.uncontracted_vars
        size_dict = {}
        inputs = []
        for tensorRV in tensorRV_list:
            inputs.append(tuple(tensorRV.variables))
            for var in tensorRV.variables:
                size_dict[var] = 4
        output = tuple(uncontracted_vars)
        opt = ctg.HyperOptimizer()
        tree = opt.search(inputs, output, size_dict)
        # print(tree)
        # print(tree.contraction_width(), tree.contraction_cost())
        path = tree.get_path()
        assert len(path) == len(tensorRV_list) - 1, "The number of contractions should be equal to the number of tensors minus 1."
        
        for i, j in path:
            new_tensorRV = tensorRV_list[i] @ tensorRV_list[j]
            if i > j:
                del tensorRV_list[i]
                del tensorRV_list[j]
            else:
                del tensorRV_list[j]
                del tensorRV_list[i]
            tensorRV_list.append(new_tensorRV)
        
        assert len(tensorRV_list) == 1, "The number of tensors should be 1 after contraction."
        return tensorRV_list[0]
    
    def contract_tensors(self, tensorRV_list=None, uncontracted_vars=None, obs=None):
        if obs == None:
            return self._contract_tensors(tensorRV_list, uncontracted_vars)
        else:
            if tensorRV_list is None:
                tensorRV_list = self.tensorRV_list
            if uncontracted_vars is None:
                uncontracted_vars = self.uncontracted_vars
            ret = []
            for ob in tqdm(obs):
                tensorRV_list_with_ob = []
                for i, uncontracted_var in enumerate(self.uncontracted_vars):
                    if ob[i] == 'I':
                        tensorRV_list_with_ob.append(TensorRV(np.array([1, 0, 0, 0]), np.zeros([4, 4]), [uncontracted_var]))
                    elif ob[i] == 'X':
                        tensorRV_list_with_ob.append(TensorRV(np.array([0, 1, 0, 0]), np.zeros([4, 4]), [uncontracted_var]))
                    elif ob[i] == 'Y':
                        tensorRV_list_with_ob.append(TensorRV(np.array([0, 0, 1, 0]), np.zeros([4, 4]), [uncontracted_var]))
                    elif ob[i] == 'Z':
                        tensorRV_list_with_ob.append(TensorRV(np.array([0, 0, 0, 1]), np.zeros([4, 4]), [uncontracted_var]))
                    else:
                        raise ValueError("Invalid observable.")
                tensorRV = self._contract_tensors(tensorRV_list + tensorRV_list_with_ob, [])
                ret.append(tensorRV)
            return ret

    def estimated_ground_state_energy(self, hamiltonian: SparsePauliOp) -> float:
        mole_ps = []
        for pauli in hamiltonian.paulis:
            mole_ps.append(str(pauli))
        mole_weights = []
        for coeff in hamiltonian.coeffs:
            mole_weights.append(coeff.real)
        mole_table = dict(zip(mole_ps, mole_weights))
        num_qubits = len(hamiltonian.paulis[0])
        
        var_on_mole_paulistrings = 0
        mean_on_mole_paulistrings = 0
        print('Contracting tensors...')
        start_time = time.time()
        tensorRV_results = self.contract_tensors(obs=mole_ps)
        end_time = time.time()
        print(f'Tensors contracted. Time taken: {end_time - start_time} seconds')
        for tensorRV, ps in zip(tensorRV_results, mole_ps):
            # tensorRV are all scalar in this case when you put 'obs=mole_ps' in 'contract_tensors'
            weight = mole_table[ps]
            var_on_mole_paulistrings += tensorRV.variance_of_trace_as_scalar(num_qubits) * weight * weight
            # calculate mean on molecule paulistrings
            mean_on_mole_paulistrings += tensorRV.mean_of_trace_as_scalar(num_qubits) * mole_table[ps]
        # mu minus 3 sigma is the estimated lowest energy
        estimated_lowest_energy = mean_on_mole_paulistrings - 3 * np.sqrt(var_on_mole_paulistrings)
        return estimated_lowest_energy

    def find_models(self):
        """
        Find all non-zero weighted models(Pauli strings) in the final density matrix state.
        
        Args:
            tensor_list: List of tensors to contract.
            uncontracted_vars: List of uncontracted variables.
        
        Returns:
            A tensor representing the result of the contraction.
        """
        tensorBoolean_list = self.tensorBoolean_list
        uncontracted_vars = self.uncontracted_vars
        size_dict = {}
        inputs = []
        for tensorBoolean in tensorBoolean_list:
            inputs.append(tuple(tensorBoolean.variables))
            for var in tensorBoolean.variables:
                size_dict[var] = 4
        output = tuple(uncontracted_vars)
        opt = ctg.HyperOptimizer()
        tree = opt.search(inputs, output, size_dict)
        print(tree)
        print(tree.contraction_width(), tree.contraction_cost())
        path = tree.get_path()
        assert len(path) == len(tensorBoolean_list) - 1, "The number of contractions should be equal to the number of tensors minus 1."
        
        for i, j in path:
            new_tensorBoolean = tensorBoolean_list[i] @ tensorBoolean_list[j]
            if i > j:
                del tensorBoolean_list[i]
                del tensorBoolean_list[j]
            else:
                del tensorBoolean_list[j]
                del tensorBoolean_list[i]
            tensorBoolean_list.append(new_tensorBoolean)
        
        assert len(tensorBoolean_list) == 1, "The number of tensors should be 1 after contraction."
        return tensorBoolean_list[0]
    
        
    
    def find_models2(self, true_false_network=None, uncontracted_vars=None):
        """
        Find all non-zero weighted models(Pauli strings) in the final density matrix state.
        
        Args:
            tensor_list: List of tensors to contract.
            uncontracted_vars: List of uncontracted variables.
        
        Returns:
            A tensor representing the result of the contraction.
        """
        if true_false_network is None:
            true_false_network = self.true_false_network
        if uncontracted_vars is None:
            uncontracted_vars = self.uncontracted_vars
        size_dict = {}
        inputs = []
        inputs_shape = []
        inputs_size = []
        arrays = []
        for factor in true_false_network.factors:
            for var in factor.variables:
                size_dict[var] = 4
            tensor_vars = factor.variables
            inputs.append(tuple(tensor_vars))
            inputs_shape.append(tuple([4 for var in tensor_vars]))
            size = 1
            for var in tensor_vars:
                size = size * 4
            inputs_size.append(size)
            arrays.append(factor.values.reshape(inputs_shape[-1]))
        output = tuple(uncontracted_vars)
        tn = qtn.TensorNetwork([
            qtn.Tensor(array, inds)
            for array, inds in zip(arrays, inputs)
        ])

        tensor = tn.contract(..., optimize=ctg.HyperOptimizer())

        return tensor
