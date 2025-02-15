import sys

import galois
import numpy as np
from scipy.linalg import block_diag

sys.path.append('../aes_numpy')
from aes_1d import MyAES2  # type: ignore
from aes_tables import shift_table, shift_table_inv, mix_matrix  # type: ignore
from utils import timer


def uint8_to_uint4(arr: np.ndarray) -> np.ndarray:
    """
    将uint8数组转化为uint4数组

    :param arr: uint8数组
    :return: uint4数组
    :example: (9, 4, 4) -> (9, 4, 8)
    """

    return np.stack((arr >> 4, arr & 0xf), axis=-1).reshape(arr.shape[:-1] + (-1,))


def uint4_to_uint8(arr: np.ndarray) -> np.ndarray:
    """
    将uint4数组转化为uint8数组

    :param arr: uint4数组
    :return: uint8数组
    :example: (9, 4, 8) -> (9, 4, 4)
    """

    arr = arr.reshape(arr.shape[:-1] + (-1, 2))
    return arr[..., 0] << 4 | arr[..., 1]


def uint4_to_uint1(arr: np.ndarray) -> np.ndarray:
    """
    将uint4数组转化为uint1数组

    :param arr: uint4数组
    :return: uint1数组
    :example: (9, 4, 8) -> (9, 4, 32)
    """

    return np.unpackbits(arr[..., np.newaxis], axis=-1, bitorder='little', count=4)[..., ::-1].reshape(arr.shape[:-1] + (-1,))


def uint1_to_uint4(arr: np.ndarray) -> np.ndarray:
    """
    将uint1数组转化为uint4数组

    :param arr: uint1数组
    :return: uint4数组
    :example: (9, 4, 32) -> (9, 4, 8)
    """

    return np.packbits(arr.reshape(arr.shape[:-1] + (-1, 4))[..., ::-1], axis=-1, bitorder='little').reshape(arr.shape[:-1] + (-1,))


def shift_rows_with_broadcast(arr: np.ndarray):
    """
    对输入的numpy数组的最后一维进行行移位操作

    :param arr: numpy数组
    :return: None
    :example: (10, 16) -> (10, 16)
    """

    arr[:] = arr[..., shift_table]


def get_bijections(*shape: int) -> np.ndarray:
    """
    生成指定形状的4位随机置换数组

    :param shape: 数组形状
    :return: 生成的4位随机置换数组
    """

    arr = np.empty(shape + (16,), dtype=np.uint8)
    for index in np.ndindex(shape):
        arr[index] = rng.permutation(16)
    return arr


def get_mixing_bijections_and_inv(shape: tuple, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """
    生成GF(2)上指定形状的维度为dim的随机矩阵数组和其逆矩阵数组

    :param shape: 数组形状
    :param dim: 矩阵维度
    :return: 随机矩阵数组和其逆矩阵数组
    """

    arr = np.empty(shape + (dim, dim), dtype=np.uint8)
    arr_inv = np.empty(shape + (dim, dim), dtype=np.uint8)
    for index in np.ndindex(shape):
        while True:
            matrix = galois.GF2.Random((dim, dim))
            if np.linalg.det(matrix) != 0:
                arr[index] = matrix.view(np.ndarray)
                arr_inv[index] = np.linalg.inv(matrix).view(np.ndarray)
                break
    return arr, arr_inv


def mixing_bijection(matrix: np.ndarray, arr: np.ndarray) -> np.ndarray:
    """
    对输入的矩阵数组和uint4数组进行矩阵乘法并返回uint4数组

    :param matrix: GF(2)上的矩阵数组
    :param arr: uint4数组
    :return: 矩阵乘法后的uint4数组
    """

    arr = uint4_to_uint1(arr)
    arr = matrix @ arr[..., np.newaxis] % 2
    arr = uint1_to_uint4(arr.reshape(arr.shape[:-2] + (-1,)))
    return arr


rng = np.random.default_rng()
GF = galois.GF(2 ** 8, irreducible_poly='x^8 + x^4 + x^3 + x + 1')
mix_matrix_GF = mix_matrix.view(GF)


class WBAES:
    def __init__(self, key: bytes):
        self.key = key
        self.round_keys = MyAES2.get_round_keys(key)

        with timer('T_boxes', 'Generating T_boxes...'):
            self.T_boxes = self.get_T_boxes()

        with timer('Tyi_tables', 'Generating Tyi_tables...'):
            self.Tyi_tables = WBAES.get_Tyi_tables()

        with timer('xor_tables', 'Generating xor_tables...'):
            self.xor_tables = WBAES.get_xor_tables()

        with timer('T_Tyi_tables', 'Generating T_Tyi_tables...'):
            self.T_Tyi_tables = self.get_T_Tyi_tables()

        T_Tyi_bijections = get_bijections(9, 16, 8)
        T_Tyi_bijections_inv = np.argsort(T_Tyi_bijections).astype(np.uint8)
        xor_bijections = get_bijections(9, 4, 3, 8)
        xor_bijections_inv = np.argsort(xor_bijections).astype(np.uint8)

        with timer('T_Tyi_tables_with_encodings', 'Generating T_Tyi_tables_with_encodings...'):
            self.T_Tyi_tables_with_encodings = self.get_T_Tyi_tables_with_encodings(T_Tyi_bijections, xor_bijections_inv)

        with timer('last_T_box_with_encodings', 'Generating last_T_box_with_encodings...'):
            self.last_T_box_with_encodings = self.get_last_T_box_with_encodings(xor_bijections_inv)

        with timer('xor_tables_with_encodings', 'Generating xor_tables_with_encodings...'):
            self.xor_tables_with_encodings = self.get_xor_tables_with_encodings(T_Tyi_bijections_inv, xor_bijections, xor_bijections_inv)

        MBs, MBs_inv = get_mixing_bijections_and_inv((9, 4), 32)
        Lis, Lis_inv = get_mixing_bijections_and_inv((9, 16), 8)

        with timer('Li_inv_T_Tyi_MB_tables', 'Generating Li_inv_T_Tyi_MB_tables...'):
            self.Li_inv_T_Tyi_MB_tables = self.get_Li_inv_T_Tyi_MB_tables(Lis_inv, MBs)

        with timer('MBi_inv_L_tables', 'Generating MBi_inv_L_tables...'):
            self.MBi_inv_L_tables = WBAES.get_MBi_inv_L_tables(Lis, MBs_inv)

        with timer('Li_inv_last_T_box', 'Generating Li_inv_last_T_box...'):
            self.Li_inv_last_T_box = self.get_Li_inv_last_T_box(Lis_inv)

        with timer('xor_tables2', 'Generating xor_tables2...'):
            self.xor_tables2 = WBAES.get_xor_tables()

    def get_T_boxes(self) -> np.ndarray:
        """
        生成T_boxes数组，形状为(10, 16, 256)

        :return: T_boxes数组
        """

        round_keys = self.round_keys.copy()
        shift_rows_with_broadcast(round_keys[:-1])
        T_boxes = np.arange(256, dtype=np.uint8) ^ round_keys[:-1, :, np.newaxis]
        MyAES2.sub_bytes(T_boxes)
        T_boxes[-1] ^= round_keys[-1, :, np.newaxis]
        return T_boxes

    @staticmethod
    def get_Tyi_tables() -> np.ndarray:
        """
        生成Tyi_tables数组，形状为(256, 4, 8)

        :return: Tyi_tables数组
        """

        Tyi_tables = np.arange(256, dtype=np.uint8).reshape(256, 1, 1).view(GF) * mix_matrix_GF.T
        Tyi_tables = uint8_to_uint4(Tyi_tables.view(np.ndarray))
        return Tyi_tables

    @staticmethod
    def get_xor_tables() -> np.ndarray:
        """
        生成xor_tables数组，形状为(9, 4, 3, 8, 16, 16)

        :return: xor_tables数组
        """

        xor_tables = np.fromfunction(np.bitwise_xor, (16, 16), dtype=np.uint8)
        xor_tables = np.broadcast_to(xor_tables, (9, 4, 3, 8, 16, 16))
        return xor_tables

    def get_T_Tyi_tables(self) -> np.ndarray:
        """
        生成T_Tyi_tables数组，形状为(9, 16, 256, 8)

        :return: T_Tyi_tables
        """

        T_Tyi_tables = self.Tyi_tables[self.T_boxes[:-1], np.tile(np.arange(4), 4).reshape(16, 1)]
        return T_Tyi_tables

    def encrypt(self, plaintext: bytes) -> bytes:
        """
        使用T_Tyi_tables, xor_tables, T_boxes[-1]数组加密

        :param plaintext: 明文字节串
        :return: 密文字节串
        """

        state = np.frombuffer(bytearray(plaintext), dtype=np.uint8)
        for r in range(9):
            MyAES2.shift_rows(state)
            arr = self.T_Tyi_tables[r, np.arange(16), state]
            xor1 = self.xor_tables[r, np.arange(4).reshape(4, 1), 0, np.arange(8), arr[np.arange(4) * 4 + 0], arr[np.arange(4) * 4 + 1]]
            xor2 = self.xor_tables[r, np.arange(4).reshape(4, 1), 1, np.arange(8), arr[np.arange(4) * 4 + 2], arr[np.arange(4) * 4 + 3]]
            xor3 = self.xor_tables[r, np.arange(4).reshape(4, 1), 2, np.arange(8), xor1, xor2]
            state[:] = uint4_to_uint8(xor3).ravel()
        MyAES2.shift_rows(state)
        state[:] = self.T_boxes[-1, np.arange(16), state]
        return state.tobytes()

    def get_T_Tyi_tables_with_encodings(self, T_Tyi_bijections: np.ndarray, xor_bijections_inv: np.ndarray) -> np.ndarray:
        """
        生成T_Tyi_tables_with_encodings数组，形状为(9, 16, 256, 8)

        :return: T_Tyi_tables_with_encodings
        """

        arr = np.arange(256, dtype=np.uint8)
        arr1 = np.concatenate((
            np.broadcast_to(arr, (1, 16, 256)),
            xor_bijections_inv[np.arange(8).reshape(8, 1, 1), (shift_table // 4).reshape(16, 1), 2, (shift_table % 4 * 2 + 0).reshape(16, 1), arr >> 4] << 4 |
            xor_bijections_inv[np.arange(8).reshape(8, 1, 1), (shift_table // 4).reshape(16, 1), 2, (shift_table % 4 * 2 + 1).reshape(16, 1), arr & 0xf]
        ))
        arr2 = self.T_Tyi_tables[np.arange(9).reshape(9, 1, 1), np.arange(16).reshape(16, 1), arr1]
        T_Tyi_tables_with_encodings = T_Tyi_bijections[np.arange(9).reshape(9, 1, 1, 1), np.arange(16).reshape(16, 1, 1), np.arange(8), arr2]
        return T_Tyi_tables_with_encodings

    def get_last_T_box_with_encodings(self, xor_bijections_inv: np.ndarray) -> np.ndarray:
        """
        生成last_T_box_with_encodings数组，形状为(16, 256)

        :return: last_T_box_with_encodings
        """

        arr = np.arange(256, dtype=np.uint8)
        arr1 = xor_bijections_inv[-1, (shift_table // 4).reshape(16, 1), 2, (shift_table % 4 * 2 + 0).reshape(16, 1), arr >> 4] << 4 | xor_bijections_inv[-1, (shift_table // 4).reshape(16, 1), 2, (shift_table % 4 * 2 + 1).reshape(16, 1), arr & 0xf]
        last_T_box_with_encodings = self.T_boxes[-1, np.arange(16).reshape(16, 1), arr1]
        return last_T_box_with_encodings

    def get_xor_tables_with_encodings(self, T_Tyi_bijections_inv: np.ndarray, xor_bijections: np.ndarray, xor_bijections_inv: np.ndarray) -> np.ndarray:
        """
        生成xor_tables_with_encodings数组，形状为(9, 4, 3, 8, 16, 16)

        :return: xor_tables_with_encodings
        """

        arr1 = np.stack((
            T_Tyi_bijections_inv[np.arange(9).reshape(9, 1, 1), (np.arange(4) * 4 + 0).reshape(4, 1), np.arange(8)],
            T_Tyi_bijections_inv[np.arange(9).reshape(9, 1, 1), (np.arange(4) * 4 + 2).reshape(4, 1), np.arange(8)],
            xor_bijections_inv[np.arange(9).reshape(9, 1, 1), np.arange(4).reshape(4, 1), 0, np.arange(8)]
        ))
        arr2 = np.stack((
            T_Tyi_bijections_inv[np.arange(9).reshape(9, 1, 1), (np.arange(4) * 4 + 1).reshape(4, 1), np.arange(8)],
            T_Tyi_bijections_inv[np.arange(9).reshape(9, 1, 1), (np.arange(4) * 4 + 3).reshape(4, 1), np.arange(8)],
            xor_bijections_inv[np.arange(9).reshape(9, 1, 1), np.arange(4).reshape(4, 1), 1, np.arange(8)]
        ))
        arr3 = self.xor_tables[np.arange(9).reshape(9, 1, 1, 1, 1), np.arange(4).reshape(4, 1, 1, 1), np.arange(3).reshape(3, 1, 1, 1, 1, 1), np.arange(8).reshape(8, 1, 1), arr1[..., np.newaxis], arr2[..., np.newaxis, :]]
        xor_tables_with_encodings = xor_bijections[np.arange(9).reshape(9, 1, 1, 1, 1), np.arange(4).reshape(4, 1, 1, 1), np.arange(3).reshape(3, 1, 1, 1, 1, 1), np.arange(8).reshape(8, 1, 1), arr3].transpose(1, 2, 0, 3, 4, 5)
        return xor_tables_with_encodings

    def encrypt_with_tables_with_encodings(self, plaintext: bytes) -> bytes:
        """
        使用T_Tyi_tables_with_encodings, xor_tables_with_encodings, last_T_box_with_encodings加密

        :param plaintext: 明文字节串
        :return: 密文字节串
        """

        state = np.frombuffer(bytearray(plaintext), dtype=np.uint8)
        for r in range(9):
            MyAES2.shift_rows(state)
            arr = self.T_Tyi_tables_with_encodings[r, np.arange(16), state]
            xor1 = self.xor_tables_with_encodings[r, np.arange(4).reshape(4, 1), 0, np.arange(8), arr[np.arange(4) * 4 + 0], arr[np.arange(4) * 4 + 1]]
            xor2 = self.xor_tables_with_encodings[r, np.arange(4).reshape(4, 1), 1, np.arange(8), arr[np.arange(4) * 4 + 2], arr[np.arange(4) * 4 + 3]]
            xor3 = self.xor_tables_with_encodings[r, np.arange(4).reshape(4, 1), 2, np.arange(8), xor1, xor2]
            state[:] = uint4_to_uint8(xor3).ravel()
        MyAES2.shift_rows(state)
        state[:] = self.last_T_box_with_encodings[np.arange(16), state]
        return state.tobytes()

    def get_Li_inv_T_Tyi_MB_tables(self, Lis_inv: np.ndarray, MBs: np.ndarray) -> np.ndarray:
        """
        生成Li_inv_T_Tyi_MB_tables数组，形状为(9, 16, 256, 8)

        :param Lis_inv: GF(2)上的逆矩阵数组, 形状为(9, 16, 8, 8)
        :param MBs: GF(2)上的矩阵数组, 形状为(9, 4, 32, 32)
        :return: Li_inv_T_Tyi_MB_tables
        """

        Li_inv_T_Tyi_MB_tables = np.empty((9, 16, 256, 8), dtype=np.uint8)

        arr = np.arange(256, dtype=np.uint8)
        Li_inv_T_Tyi_MB_tables[0] = mixing_bijection(np.repeat(MBs[0, np.arange(4)], 4, axis=0)[:, np.newaxis], self.T_Tyi_tables[0, np.arange(16).reshape(16, 1), arr])

        arr1 = Lis_inv[:-1, np.arange(16)]
        arr2 = uint8_to_uint4(arr[:, np.newaxis])
        arr3 = mixing_bijection(arr1[:, :, np.newaxis], arr2)
        arr4 = uint4_to_uint8(arr3).reshape(arr3.shape[:-2] + (-1,))
        Li_inv_T_Tyi_MB_tables[1:9] = mixing_bijection(np.repeat(MBs[1:9, np.arange(4)], 4, axis=1)[:, :, np.newaxis], self.T_Tyi_tables[np.arange(1, 9).reshape(8, 1, 1), np.arange(16).reshape(16, 1), arr4])
        return Li_inv_T_Tyi_MB_tables

    @staticmethod
    def get_MBi_inv_L_tables(Lis: np.ndarray, MBs_inv: np.ndarray) -> np.ndarray:
        """
        生成MBi_inv_L_tables数组，形状为(9, 4, 4, 256, 8)

        :param Lis: GF(2)上的矩阵数组, 形状为(9, 16, 8, 8)
        :param MBs_inv: GF(2)上的逆矩阵数组, 形状为(9, 4, 32, 32)
        :return: MBi_inv_L_tables
        """

        block_diag_vectorized = np.vectorize(lambda x: block_diag(*x), signature='(a,b,b)->(c,c)')
        Ls = block_diag_vectorized(Lis[np.arange(9).reshape(9, 1, 1), shift_table_inv[np.arange(16).reshape(4, 4)]])
        arr = np.arange(256, dtype=np.uint8)
        arr1 = np.zeros((9, 4, 4, 256, 4), dtype=np.uint8)
        arr1[np.arange(9).reshape(9, 1, 1), np.arange(4).reshape(4, 1), np.arange(4), :, np.arange(4)] = arr
        arr2 = uint8_to_uint4(arr1)
        arr3 = mixing_bijection(MBs_inv[:, :, np.newaxis, np.newaxis], arr2)
        MBi_inv_L_tables = mixing_bijection(Ls[:, :, np.newaxis, np.newaxis], arr3)
        return MBi_inv_L_tables

    def get_Li_inv_last_T_box(self, Lis_inv: np.ndarray) -> np.ndarray:
        """
        生成Li_inv_last_T_box数组，形状为(16, 256)

        :param Lis_inv: GF(2)上的逆矩阵数组, 形状为(9, 16, 8, 8)
        :return: Li_inv_last_T_box
        """

        arr = np.arange(256, dtype=np.uint8)
        arr1 = uint8_to_uint4(arr[:, np.newaxis])
        arr2 = mixing_bijection(Lis_inv[-1, :, np.newaxis], arr1)
        arr3 = uint4_to_uint8(arr2).reshape(arr2.shape[:-2] + (-1,))
        Li_inv_last_T_box = self.T_boxes[-1, np.arange(16).reshape(16, 1), arr3]
        return Li_inv_last_T_box

    def encrypt_with_tables_with_mixing_bijections(self, plaintext: bytes) -> bytes:
        """
        使用Li_inv_T_Tyi_MB_tables, MBi_inv_L_tables, Li_inv_last_T_box数组加密

        :param plaintext: 明文字节串
        :return: 密文字节串
        """

        state = np.frombuffer(bytearray(plaintext), dtype=np.uint8)
        for r in range(9):
            MyAES2.shift_rows(state)

            arr = self.Li_inv_T_Tyi_MB_tables[r, np.arange(16), state]
            xor1 = self.xor_tables[r, np.arange(4).reshape(4, 1), 0, np.arange(8), arr[np.arange(4) * 4 + 0], arr[np.arange(4) * 4 + 1]]
            xor2 = self.xor_tables[r, np.arange(4).reshape(4, 1), 1, np.arange(8), arr[np.arange(4) * 4 + 2], arr[np.arange(4) * 4 + 3]]
            xor3 = self.xor_tables[r, np.arange(4).reshape(4, 1), 2, np.arange(8), xor1, xor2]
            xor3 = uint4_to_uint8(xor3).ravel()

            arr = self.MBi_inv_L_tables[r, np.repeat(np.arange(4), 4), np.tile(np.arange(4), 4), xor3]
            xor1 = self.xor_tables2[r, np.arange(4).reshape(4, 1), 0, np.arange(8), arr[np.arange(4) * 4 + 0], arr[np.arange(4) * 4 + 1]]
            xor2 = self.xor_tables2[r, np.arange(4).reshape(4, 1), 1, np.arange(8), arr[np.arange(4) * 4 + 2], arr[np.arange(4) * 4 + 3]]
            xor3 = self.xor_tables2[r, np.arange(4).reshape(4, 1), 2, np.arange(8), xor1, xor2]
            state[:] = uint4_to_uint8(xor3).ravel()

        MyAES2.shift_rows(state)
        state[:] = self.Li_inv_last_T_box[np.arange(16), state]
        return state.tobytes()


if __name__ == '__main__':
    import timeit

    key = '1234567890123456'.encode()
    plaintext = 'abcdefghijklmnop'.encode()

    wb_aes = WBAES(key)
    ciphertext1 = wb_aes.encrypt(plaintext)
    ciphertext2 = wb_aes.encrypt_with_tables_with_encodings(plaintext)
    ciphertext3 = wb_aes.encrypt_with_tables_with_mixing_bijections(plaintext)
    print(ciphertext1.hex())
    print(ciphertext2.hex())
    print(ciphertext3.hex())

    print(timeit.timeit(lambda: wb_aes.encrypt(plaintext), number=100))
    print(timeit.timeit(lambda: wb_aes.encrypt_with_tables_with_encodings(plaintext), number=100))
    print(timeit.timeit(lambda: wb_aes.encrypt_with_tables_with_mixing_bijections(plaintext), number=100))
