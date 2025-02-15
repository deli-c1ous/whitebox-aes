import sys
import random

import galois
import numpy as np
from more_itertools import sliced
from scipy.linalg import block_diag

sys.path.append('../aes')
from aes_1d import MyAES2, gf_mul  # type: ignore
from aes_tables import shift_table, shift_table_inv  # type: ignore
from utils import timer


def get_bijection() -> list[int]:
    return random.sample(range(16), 16)


def get_mixing_bijection(dim: int) -> galois.FieldArray:
    while True:
        matrix = galois.GF2.Random((dim, dim))
        if np.linalg.det(matrix) != 0:
            return matrix


def mixing_bijection(matrix: np.ndarray, x: list[int]) -> list[int]:
    x = ''.join(map(lambda nibble: bin(nibble)[2:].zfill(4), x))
    x = list(map(int, x))
    x = matrix @ np.array(x, dtype='uint8') % 2
    x = ''.join(map(str, x))
    x = list(map(lambda bits: int(bits, 2), sliced(x, 4)))
    return x


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

        T_Tyi_bijections = [[[get_bijection() for _ in range(8)] for _ in range(16)] for _ in range(9)]
        xor_bijections = [[[[get_bijection() for _ in range(8)] for _ in range(3)] for _ in range(4)] for _ in range(9)]

        with timer('T_Tyi_tables_with_encodings', 'Generating T_Tyi_tables_with_encodings...'):
            self.T_Tyi_tables_with_encodings = self.get_T_Tyi_tables_with_encodings(T_Tyi_bijections, xor_bijections)
        with timer('xor_tables_with_encodings', 'Generating xor_tables_with_encodings...'):
            self.xor_tables_with_encodings = self.get_xor_tables_with_encodings(T_Tyi_bijections, xor_bijections)
        with timer('last_T_box_with_encodings', 'Generating last_T_box_with_encodings...'):
            self.last_T_box_with_encodings = self.get_last_T_box_with_encodings(xor_bijections)

        MBs = [[get_mixing_bijection(32) for _ in range(4)] for _ in range(9)]
        Lis = [[get_mixing_bijection(8) for _ in range(16)] for _ in range(9)]

        with timer('Li_inv_T_Tyi_MB_tables', 'Generating Li_inv_T_Tyi_MB_tables...'):
            self.Li_inv_T_Tyi_MB_tables = self.get_Li_inv_T_Tyi_MB_tables(Lis, MBs)
        with timer('MBi_inv_L_tables', 'Generating MBi_inv_L_tables...'):
            self.MBi_inv_L_tables = WBAES.get_MBi_inv_L_tables(Lis, MBs)
        with timer('Li_inv_last_T_box', 'Generating Li_inv_last_T_box...'):
            self.Li_inv_last_T_box = self.get_Li_inv_last_T_box(Lis)
        with timer('xor_tables2', 'Generating xor_tables2...'):
            self.xor_tables2 = WBAES.get_xor_tables()

    def get_T_boxes(self) -> list:
        T_boxes = [[[0 for _ in range(256)] for _ in range(16)] for _ in range(10)]
        for r in range(10):
            for x in range(256):
                round_key = self.round_keys[r]
                state = [x] * 16
                state = MyAES2.add_round_key(state, MyAES2.shift_rows(round_key))
                state = MyAES2.sub_bytes(state)
                if r == 9: state = MyAES2.add_round_key(state, self.round_keys[10])
                for i in range(16): T_boxes[r][i][x] = state[i]
        return T_boxes

    @staticmethod
    def get_Tyi_tables() -> list:
        Tyi_tables = [[[[[0 for _ in range(8)] for _ in range(256)] for _ in range(4)] for _ in range(4)] for _ in range(9)]
        for r in range(9):
            for i in range(4):
                for x in range(256):
                    a = bytes([gf_mul(x, 2), gf_mul(x, 1), gf_mul(x, 1), gf_mul(x, 3)]).hex()
                    b = bytes([gf_mul(x, 3), gf_mul(x, 2), gf_mul(x, 1), gf_mul(x, 1)]).hex()
                    c = bytes([gf_mul(x, 1), gf_mul(x, 3), gf_mul(x, 2), gf_mul(x, 1)]).hex()
                    d = bytes([gf_mul(x, 1), gf_mul(x, 1), gf_mul(x, 3), gf_mul(x, 2)]).hex()
                    Tyi_tables[r][i][0][x] = list(map(lambda x: int(x, 16), a))
                    Tyi_tables[r][i][1][x] = list(map(lambda x: int(x, 16), b))
                    Tyi_tables[r][i][2][x] = list(map(lambda x: int(x, 16), c))
                    Tyi_tables[r][i][3][x] = list(map(lambda x: int(x, 16), d))
        return Tyi_tables

    @staticmethod
    def get_xor_tables() -> list:
        xor_tables = [[[[[[i ^ j for i in range(16)] for j in range(16)] for _ in range(8)] for _ in range(3)] for _ in range(4)] for _ in range(9)]
        return xor_tables

    def get_T_Tyi_tables(self) -> list:
        T_Tyi_tables = [[[[[0 for _ in range(8)] for _ in range(256)] for _ in range(4)] for _ in range(4)] for _ in range(9)]
        for r in range(9):
            for i in range(16):
                for x in range(256):
                    v = self.T_boxes[r][i][x]
                    T_Tyi_tables[r][i // 4][i % 4][x] = self.Tyi_tables[r][i // 4][i % 4][v]
        return T_Tyi_tables

    def encrypt(self, plaintext: bytes) -> bytes:
        state = list(plaintext)
        for r in range(9):
            state = MyAES2.shift_rows(state)
            for i in range(4):
                a = self.T_Tyi_tables[r][i][0][state[i * 4 + 0]]
                b = self.T_Tyi_tables[r][i][1][state[i * 4 + 1]]
                c = self.T_Tyi_tables[r][i][2][state[i * 4 + 2]]
                d = self.T_Tyi_tables[r][i][3][state[i * 4 + 3]]

                xor1 = [self.xor_tables[r][i][0][j][a[j]][b[j]] for j in range(8)]
                xor2 = [self.xor_tables[r][i][1][j][c[j]][d[j]] for j in range(8)]
                xor3 = [self.xor_tables[r][i][2][j][xor1[j]][xor2[j]] for j in range(8)]

                xor3 = ''.join(map(lambda x: hex(x)[2:], xor3))
                xor3 = bytes.fromhex(xor3)

                state[i * 4 + 0] = xor3[0]
                state[i * 4 + 1] = xor3[1]
                state[i * 4 + 2] = xor3[2]
                state[i * 4 + 3] = xor3[3]

        state = MyAES2.shift_rows(state)
        ciphertext = bytes(map(lambda i: self.T_boxes[-1][i][state[i]], range(16)))
        return ciphertext

    def get_T_Tyi_tables_with_encodings(self, T_Tyi_bijections: list, xor_bijections: list) -> list:
        T_Tyi_tables_with_encodings = [[[[[0 for _ in range(8)] for _ in range(256)] for _ in range(4)] for _ in range(4)] for _ in range(9)]
        for r in range(9):
            for i in range(16):
                for x in range(256):
                    new_x = x
                    if r != 0:
                        new_x = hex(x)[2:].zfill(2)
                        index = shift_table[i] * 2
                        new_x1 = xor_bijections[r - 1][index // 8][2][index % 8].index(int(new_x[0], 16))
                        new_x2 = xor_bijections[r - 1][index // 8][2][index % 8 + 1].index(int(new_x[1], 16))
                        new_x = hex(new_x1)[2:] + hex(new_x2)[2:]
                        new_x = int(new_x, 16)
                    a = self.T_Tyi_tables[r][i // 4][i % 4][new_x]
                    a = list(map(lambda bijection, nibble: bijection[nibble], T_Tyi_bijections[r][i], a))
                    T_Tyi_tables_with_encodings[r][i // 4][i % 4][x] = a
        return T_Tyi_tables_with_encodings

    def get_xor_tables_with_encodings(self, T_Tyi_bijections: list, xor_bijections: list) -> list:
        xor_tables_with_encodings = [[[[[[0 for _ in range(16)] for _ in range(16)] for _ in range(8)] for _ in range(3)] for _ in range(4)] for _ in range(9)]
        for r in range(9):
            for i in range(4):
                for j in range(8):
                    for x in range(16):
                        for y in range(16):
                            new_x = T_Tyi_bijections[r][i * 4 + 0][j].index(x)
                            new_y = T_Tyi_bijections[r][i * 4 + 1][j].index(y)
                            a = self.xor_tables[r][i][0][j][new_x][new_y]
                            xor_tables_with_encodings[r][i][0][j][x][y] = xor_bijections[r][i][0][j][a]

                            new_x = T_Tyi_bijections[r][i * 4 + 2][j].index(x)
                            new_y = T_Tyi_bijections[r][i * 4 + 3][j].index(y)
                            a = self.xor_tables[r][i][1][j][new_x][new_y]
                            xor_tables_with_encodings[r][i][1][j][x][y] = xor_bijections[r][i][1][j][a]

                            new_x = xor_bijections[r][i][0][j].index(x)
                            new_y = xor_bijections[r][i][1][j].index(y)
                            a = self.xor_tables[r][i][2][j][new_x][new_y]
                            xor_tables_with_encodings[r][i][2][j][x][y] = xor_bijections[r][i][2][j][a]
        return xor_tables_with_encodings

    def get_last_T_box_with_encodings(self, xor_bijections: list) -> list:
        last_T_box_with_encodings = [[0 for _ in range(256)] for _ in range(16)]
        for i in range(16):
            for x in range(256):
                new_x = hex(x)[2:].zfill(2)
                index = shift_table[i] * 2
                new_x1 = xor_bijections[-1][index // 8][2][index % 8].index(int(new_x[0], 16))
                new_x2 = xor_bijections[-1][index // 8][2][index % 8 + 1].index(int(new_x[1], 16))
                new_x = hex(new_x1)[2:] + hex(new_x2)[2:]
                new_x = int(new_x, 16)
                last_T_box_with_encodings[i][x] = self.T_boxes[-1][i][new_x]
        return last_T_box_with_encodings

    def encrypt_with_tables_with_encodings(self, plaintext: bytes) -> bytes:
        state = list(plaintext)
        for r in range(9):
            state = MyAES2.shift_rows(state)
            for i in range(4):
                a = self.T_Tyi_tables_with_encodings[r][i][0][state[i * 4 + 0]]
                b = self.T_Tyi_tables_with_encodings[r][i][1][state[i * 4 + 1]]
                c = self.T_Tyi_tables_with_encodings[r][i][2][state[i * 4 + 2]]
                d = self.T_Tyi_tables_with_encodings[r][i][3][state[i * 4 + 3]]

                xor1 = [self.xor_tables_with_encodings[r][i][0][j][a[j]][b[j]] for j in range(8)]
                xor2 = [self.xor_tables_with_encodings[r][i][1][j][c[j]][d[j]] for j in range(8)]
                xor3 = [self.xor_tables_with_encodings[r][i][2][j][xor1[j]][xor2[j]] for j in range(8)]

                xor3 = ''.join(map(lambda x: hex(x)[2:], xor3))
                xor3 = bytes.fromhex(xor3)

                state[i * 4 + 0] = xor3[0]
                state[i * 4 + 1] = xor3[1]
                state[i * 4 + 2] = xor3[2]
                state[i * 4 + 3] = xor3[3]

        state = MyAES2.shift_rows(state)
        ciphertext = bytes(map(lambda i: self.last_T_box_with_encodings[i][state[i]], range(16)))
        return ciphertext

    def get_Li_inv_T_Tyi_MB_tables(self, Lis: list[list[np.ndarray]], MBs: list[list[np.ndarray]]) -> list:
        Li_inv_T_Tyi_MB_tables = [[[[0 for _ in range(8)] for _ in range(256)] for _ in range(16)] for _ in range(9)]
        for r in range(9):
            for i in range(16):
                MB = MBs[r][i // 4]
                if r != 0: Li_inv = np.linalg.inv(Lis[r - 1][i])
                for x in range(256):
                    new_x = x
                    if r != 0:
                        new_x = hex(new_x)[2:].zfill(2)
                        new_x = list(map(lambda nibble: int(nibble, 16), new_x))
                        new_x = mixing_bijection(Li_inv.view(np.ndarray), new_x)
                        new_x = ''.join(map(lambda x: hex(x)[2:], new_x))
                        new_x = int(new_x, 16)
                    Li_inv_T_Tyi_MB_tables[r][i][x] = mixing_bijection(MB.view(np.ndarray), self.T_Tyi_tables[r][i // 4][i % 4][new_x])
        return Li_inv_T_Tyi_MB_tables

    @staticmethod
    def get_MBi_inv_L_tables(Lis: list[list[np.ndarray]], MBs: list[list[np.ndarray]]) -> list:
        MBi_inv_L_tables = [[[[[0 for _ in range(8)] for _ in range(256)] for _ in range(4)] for _ in range(4)] for _ in range(9)]
        for r in range(9):
            for i in range(4):
                MB_inv = np.linalg.inv(MBs[r][i])
                L0 = Lis[r][shift_table_inv[i * 4 + 0]]
                L1 = Lis[r][shift_table_inv[i * 4 + 1]]
                L2 = Lis[r][shift_table_inv[i * 4 + 2]]
                L3 = Lis[r][shift_table_inv[i * 4 + 3]]
                L = block_diag(L0, L1, L2, L3)
                for j in range(4):
                    vector = [0] * 4
                    for x in range(256):
                        vector[j] = x
                        new_vector = bytes(vector).hex()
                        new_vector = list(map(lambda x: int(x, 16), new_vector))
                        new_vector = mixing_bijection(MB_inv.view(np.ndarray), new_vector)
                        new_vector = mixing_bijection(L, new_vector)
                        MBi_inv_L_tables[r][i][j][x] = new_vector
        return MBi_inv_L_tables

    def get_Li_inv_last_T_box(self, Lis: list[list[np.ndarray]]) -> list:
        Li_inv_last_T_box = [[0 for _ in range(256)] for _ in range(16)]
        for i in range(16):
            Li_inv = np.linalg.inv(Lis[-1][i])
            for x in range(256):
                new_x = hex(x)[2:].zfill(2)
                new_x = list(map(lambda x: int(x, 16), new_x))
                new_x = mixing_bijection(Li_inv.view(np.ndarray), new_x)
                new_x = ''.join(map(lambda x: hex(x)[2:], new_x))
                new_x = int(new_x, 16)
                Li_inv_last_T_box[i][x] = self.T_boxes[-1][i][new_x]
        return Li_inv_last_T_box

    def encrypt_with_tables_with_mixing_bijections(self, plaintext: bytes) -> bytes:
        state = list(plaintext)
        for r in range(9):
            state = MyAES2.shift_rows(state)
            for i in range(4):
                a = self.Li_inv_T_Tyi_MB_tables[r][i * 4 + 0][state[i * 4 + 0]]
                b = self.Li_inv_T_Tyi_MB_tables[r][i * 4 + 1][state[i * 4 + 1]]
                c = self.Li_inv_T_Tyi_MB_tables[r][i * 4 + 2][state[i * 4 + 2]]
                d = self.Li_inv_T_Tyi_MB_tables[r][i * 4 + 3][state[i * 4 + 3]]

                xor1 = [self.xor_tables[r][i][0][j][a[j]][b[j]] for j in range(8)]
                xor2 = [self.xor_tables[r][i][1][j][c[j]][d[j]] for j in range(8)]
                xor3 = [self.xor_tables[r][i][2][j][xor1[j]][xor2[j]] for j in range(8)]

                xor3 = ''.join(map(lambda x: hex(x)[2:], xor3))
                xor3 = bytes.fromhex(xor3)

                a = self.MBi_inv_L_tables[r][i][0][xor3[0]]
                b = self.MBi_inv_L_tables[r][i][1][xor3[1]]
                c = self.MBi_inv_L_tables[r][i][2][xor3[2]]
                d = self.MBi_inv_L_tables[r][i][3][xor3[3]]

                xor1 = [self.xor_tables2[r][i][0][j][a[j]][b[j]] for j in range(8)]
                xor2 = [self.xor_tables2[r][i][1][j][c[j]][d[j]] for j in range(8)]
                xor3 = [self.xor_tables2[r][i][2][j][xor1[j]][xor2[j]] for j in range(8)]

                xor3 = ''.join(map(lambda x: hex(x)[2:], xor3))
                xor3 = bytes.fromhex(xor3)

                state[i * 4 + 0] = xor3[0]
                state[i * 4 + 1] = xor3[1]
                state[i * 4 + 2] = xor3[2]
                state[i * 4 + 3] = xor3[3]

        state = MyAES2.shift_rows(state)
        ciphertext = bytes(map(lambda i: self.Li_inv_last_T_box[i][state[i]], range(16)))
        return ciphertext


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
