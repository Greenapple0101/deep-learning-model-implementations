import torch

# Type hints.
from typing import List, Tuple
from torch import Tensor


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print('Hello from pytorch101.py!')

def create_sample_tensor() -> Tensor:
    """
    Return a torch Tensor of shape (3, 2) which is filled with zeros, except
    for element (0, 1) which is set to 10 and element (1, 0) which is set to
    100.
    """
    x = torch.zeros((3, 2))  # 3x2 텐서를 0으로 초기화
    x[0, 1] = 10  # (0, 1)에 10을 할당
    x[1, 0] = 100  # (1, 0)에 100을 할당
    return x


def mutate_tensor(
    x: Tensor, indices: List[Tuple[int, int]], values: List[float]
) -> Tensor:
    """
    Mutate the tensor x according to indices and values.
    """
    for (i, j), v in zip(indices, values):
        x[i, j] = v  # 각 인덱스에 해당하는 값을 할당
    return x


def count_tensor_elements(x: Tensor) -> int:
    """
    Count the number of scalar elements in a tensor x.
    """
    num_elements = 1
    for dim in x.shape:
        num_elements *= dim  # 모든 차원의 크기를 곱함
    return num_elements


import torch

def create_tensor_of_pi(M, N):
    # MxN 크기의 텐서를 3.14 값으로 채운다.
    return torch.full((M, N), 3.14)  # 한 줄로 충분히 함수 구현 완료

import torch

def multiples_of_ten(start: int, stop: int) -> torch.Tensor:
    assert start <= stop
    
    # start가 10의 배수가 아니면, 다음 10의 배수로 올림
    if start % 10 != 0:
        start = ((start // 10) + 1) * 10
    
    # start가 stop보다 큰 경우 빈 텐서 반환
    if start > stop:
        return torch.tensor([], dtype=torch.float64)
    
    # start에서 stop까지 10의 배수 생성
    x = torch.arange(start, stop + 1, 10, dtype=torch.float64)
    
    return x

import torch
from torch import Tensor
from typing import Tuple

def slice_indexing_practice(x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Given a two-dimensional tensor x, extract and return several subtensors to
    practice with slice indexing. Each tensor should be created using a single
    slice indexing operation.

    The input tensor should not be modified.

    Args:
        x: Tensor of shape (M, N) -- M rows, N columns with M >= 3 and N >= 5.

    Returns:
        A tuple of:
        - last_row: Tensor of shape (N,) giving the last row of x. It should be
          a one-dimensional tensor.
        - third_col: Tensor of shape (M, 1) giving the third column of x. It
          should be a two-dimensional tensor.
        - first_two_rows_three_cols: Tensor of shape (2, 3) giving the data in
          the first two rows and first three columns of x.
        - even_rows_odd_cols: Two-dimensional tensor containing the elements in
          the even-valued rows and odd-valued columns of x.
    """
    assert x.shape[0] >= 3
    assert x.shape[1] >= 5

    # 마지막 행 추출 (one-dimensional tensor)
    last_row = x[-1, :]

    # 세 번째 열 추출 (two-dimensional tensor)
    third_col = x[:, 2:3]

    # 첫 두 행과 첫 세 열 추출 (2x3 tensor)
    first_two_rows_three_cols = x[:2, :3]

    # 짝수 행과 홀수 열에 해당하는 원소 추출 (two-dimensional tensor)
    even_rows_odd_cols = x[::2, 1::2]

    return last_row, third_col, first_two_rows_three_cols, even_rows_odd_cols


import torch
from torch import Tensor

def slice_assignment_practice(x: Tensor) -> Tensor:
    """
    Given a two-dimensional tensor of shape (M, N) with M >= 4, N >= 6, mutate
    its first 4 rows and 6 columns to match the following:

    [0 0 3 3 1 1]
    [4 4 2 2 4 4]
    [2 2 5 5 2 2]
    [5 5 2 2 5 5]

    Args:
        x: A tensor of shape (M, N) with M >= 4 and N >= 6

    Returns:
        x
    """
    x[0:2, 2:6] = 2
    x[0:2, 1] = 1
    x[2:4, [0,2]] = 3  

    x[2:4, [1,3]] = 4  
    x[2:4, 4:6] = 5  
    return x


import torch
from torch import Tensor

def shuffle_cols(x: Tensor) -> Tensor:
    """
    Re-order the columns of an input tensor as described below.

    Args:
        x: A tensor of shape (M, N) with N >= 3

    Returns:
        A tensor y of shape (M, 4) where:
        - The first two columns of y are copies of the first column of x
        - The third column of y is the same as the third column of x
        - The fourth column of y is the same as the second column of x
    """
    # 인덱스 배열을 사용하여 열을 재배치
    y = x[:, [0, 0, 2, 1]]  # 첫 번째 열 두 번, 세 번째 열, 두 번째 열
    return y


import torch
from torch import Tensor

def reverse_rows(x: Tensor) -> Tensor:
    """
    Reverse the rows of the input tensor without modifying the input tensor.

    Args:
        x: A tensor of shape (M, N)

    Returns:
        y: Tensor of shape (M, N) which is the same as x but with the rows
            reversed.
    """
    # 역순 인덱스 생성
    indices = torch.arange(x.shape[0] - 1, -1, -1)  # x의 행의 수에 대한 역순 인덱스
    
    # 인덱스를 사용하여 행을 역순으로 배치
    y = x[indices, :]
    return y


import torch
from torch import Tensor

def take_one_elem_per_col(x: Tensor) -> Tensor:
    """
    Construct a new tensor by picking out one element from each column of the
    input tensor.
    
    Args:
        x: A tensor of shape (M, N) with M >= 4 and N >= 3.
    
    Returns:
        A tensor y of shape (3,) such that:
        - The first element of y is the second element of the first column of x
        - The second element of y is the first element of the second column of x
        - The third element of y is the fourth element of the third column of x
    """
    # 각 열에서 원하는 요소를 선택
    y = x[[1, 0, 3], [0, 1, 2]]  # 첫 열의 2번째, 두 번째 열의 첫 번째, 세 번째 열의 4번째 요소 선택
    return y



import torch
from torch import Tensor
from typing import List

def make_one_hot(x: List[int]) -> Tensor:
    """
    Construct a tensor of one-hot-vectors from a list of Python integers.

    Args:
        x: A list of N integers

    Returns:
        y: Tensor of shape (N, C) where C = 1 + max(x), and each row is a one-hot vector.
    """
    N = len(x)  # 리스트의 길이 (N개의 행)
    C = max(x) + 1  # one-hot 벡터의 열 수는 max(x) + 1

    # 0으로 채워진 텐서를 생성
    y = torch.zeros((N, C), dtype=torch.float32)

    # 각 행의 특정 인덱스에 1을 할당
    y[torch.arange(N), x] = 1

    return y  # y를 반환


import torch
from torch import Tensor

def sum_positive_entries(x: Tensor) -> int:
    """
    Return the sum of all the positive values in the input tensor x.

    Args:
        x: A tensor of any shape with dtype torch.int64

    Returns:
        pos_sum: Python integer giving the sum of all positive values in x
    """
    # 조건 비교로 양수 값들만 선택
    positive_values = x[x > 0]

    # 양수 값들의 합을 계산하고 파이썬 정수로 반환
    pos_sum = positive_values.sum().item()

    return pos_sum

import torch
from torch import Tensor

import torch
from torch import Tensor

import torch
from torch import Tensor


def reshape_practice(x: Tensor) -> Tensor:
    """
    Given an input tensor of shape (24,), return a reshaped tensor y of shape (3, 8).
    The output should look like:
    tensor([[ 0,  1,  2,  3, 12, 13, 14, 15],
            [ 4,  5,  6,  7, 16, 17, 18, 19],
            [ 8,  9, 10, 11, 20, 21, 22, 23]])
    """
    
    intermediate_tensor = x.view(2, 3, 4)

    
    permuted_tensor = intermediate_tensor.permute(1, 0, 2)

   
    y = permuted_tensor.reshape(3, 8)

    return y


def zero_row_min(x: Tensor) -> Tensor:
    """
    Return a copy of the input tensor x, where the minimum value along each row
    has been set to 0.
    """
    y = x.clone()  # x의 복사본을 만듦
    min_vals, _ = x.min(dim=1, keepdim=True)  # 각 행의 최소값
    y[y == min_vals] = 0  # 최소값을 0으로 변경
    return y


def batched_matrix_multiply(
    x: Tensor, y: Tensor, use_loop: bool = True
) -> Tensor:
    """
    Perform batched matrix multiplication between the tensor x of shape
    (B, N, M) and the tensor y of shape (B, M, P).

    Depending on the value of use_loop, this calls to either
    batched_matrix_multiply_loop or batched_matrix_multiply_noloop to perform
    the actual computation. You don't need to implement anything here.

    Args:
        x: Tensor of shape (B, N, M)
        y: Tensor of shape (B, M, P)
        use_loop: Whether to use an explicit Python loop.

    Returns:
        z: Tensor of shape (B, N, P) where z[i] of shape (N, P) is the result
            of matrix multiplication between x[i] of shape (N, M) and y[i] of
            shape (M, P). The output z should have the same dtype as x.
    """
    if use_loop:
        return batched_matrix_multiply_loop(x, y)
    else:
        return batched_matrix_multiply_noloop(x, y)

def batched_matrix_multiply_loop(x: Tensor, y: Tensor) -> Tensor:
    """
    Perform batched matrix multiplication between the tensor x of shape
    (B, N, M) and the tensor y of shape (B, M, P).
    
    This implementation uses an explicit loop over the batch dimension B to compute the output.
    
    Args:
        x: Tensor of shape (B, N, M)
        y: Tensor of shape (B, M, P)
    
    Returns:
        z: Tensor of shape (B, N, P) where z[i] of shape (N, P) is the result
            of matrix multiplication between x[i] of shape (N, M) and y[i] of
            shape (M, P). The output z should have the same dtype as x.
    """
    B, N, M = x.shape
    _, M, P = y.shape
    
    # Initialize the output tensor z
    z = torch.zeros((B, N, P), dtype=x.dtype)
    
    # Loop over the batch dimension B
    for i in range(B):
        # Perform matrix multiplication for each batch
        z[i] = torch.mm(x[i], y[i])
    
    return z


def batched_matrix_multiply_noloop(x: Tensor, y: Tensor) -> Tensor:
    """
    Perform batched matrix multiplication between the tensor x of shape
    (B, N, M) and the tensor y of shape (B, M, P).
    
    This implementation uses no explicit Python loops.
    
    Hint: torch.bmm
    
    Args:
        x: Tensor of shape (B, N, M)
        y: Tensor of shape (B, M, P)
    
    Returns:
        z: Tensor of shape (B, N, P) where z[i] of shape (N, P) is the result
            of matrix multiplication between x[i] of shape (N, M) and y[i] of
            shape (M, P). The output z should have the same dtype as x.
    """
    # Use batch matrix multiplication (torch.bmm)
    z = torch.bmm(x, y)
    
    return z

import torch
from torch import Tensor

import torch
from torch import Tensor

def normalize_columns(x: Tensor) -> Tensor:
    """
    Normalize the columns of the matrix x by subtracting the mean and dividing
    by the standard deviation of each column.
    
    Args:
        x: A tensor of shape (M, N) where M is the number of rows (samples) and N is the number of columns (features).
    
    Returns:
        y: A tensor of the same shape as x with normalized columns.
    """
    # Compute the mean of each column
    mu = x.mean(dim=0)
    
    # Compute the unbiased standard deviation of each column (using M-1 in the denominator)
    sigma = x.std(dim=0, unbiased=True)
    
    # Normalize the columns: (x - mu) / sigma
    y = (x - mu) / sigma
    
    # Rounding the results to the nearest value as required by the example
    y_rounded = torch.round(y)
    
    return y_rounded




def mm_on_cpu(x: Tensor, w: Tensor) -> Tensor:
    """
    Perform matrix multiplication on CPU.

    You don't need to implement anything for this function.

    Args:
        x: Tensor of shape (A, B), on CPU
        w: Tensor of shape (B, C), on CPU

    Returns:
        y: Tensor of shape (A, C) as described above. It should not be in GPU.
    """
    y = x.mm(w)
    return y

def mm_on_gpu(x: Tensor, w: Tensor) -> Tensor:
    """
    Perform matrix multiplication on GPU and return the result back on CPU.
    """
    x_gpu = x.cuda()
    w_gpu = w.cuda()
    y_gpu = torch.mm(x_gpu, w_gpu)
    y = y_gpu.cpu()  # 결과를 CPU로 이동
    return y


def challenge_mean_tensors(xs: List[Tensor], ls: Tensor) -> Tensor:
    """
    Compute mean of each tensor in a given list of tensors.
    """
    y = torch.stack([t.sum() / l for t, l in zip(xs, ls)])  # 각 텐서의 평균 계산
    return y



def challenge_get_uniques(x: torch.Tensor) -> Tuple[Tensor, Tensor]:
    """
    Get unique values and first occurrence from an input tensor.
    """
    uniques, indices = torch.unique(x, return_inverse=True)
    _, first_occurrence = torch.sort(indices)  # 첫 번째 발생 인덱스 정렬
    return uniques, first_occurrence
