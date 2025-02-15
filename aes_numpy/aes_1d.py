# from functools import cache

import numpy as np
import galois

from aes_2d import MyAES
from aes_tables import shift_table, shift_table_inv, mix_matrix, mix_matrix_inv, RCONs

# @np.vectorize(otypes=[np.uint8])
# @cache
# def gf_mul(a, b):
#     p = 0
#     while b:
#         if b & 1: p ^= a
#         msb = a & 0x80
#         a <<= 1
#         if msb: a ^= 0x1b
#         b >>= 1
#     return p & 0xff


GF = galois.GF(2 ** 8, irreducible_poly='x^8 + x^4 + x^3 + x + 1')
mix_matrix_GF = mix_matrix.view(GF)
mix_matrix_inv_GF = mix_matrix_inv.view(GF)


class MyAES2(MyAES):
    def __init__(self, key):
        super().__init__(key)
        self.round_keys = self.get_round_keys(key)

    @staticmethod
    def get_round_keys(key: bytes) -> np.ndarray:
        round_keys = np.empty((11, 16), dtype=np.uint8)
        round_keys[0] = np.frombuffer(key, dtype=np.uint8)
        for i in range(1, 11):
            round_keys[i, 0:4] = round_keys[i - 1, 0:4] ^ MyAES.g(round_keys[i - 1, 12:16], RCONs[i - 1])
            round_keys[i, 4:8] = round_keys[i - 1, 4:8] ^ round_keys[i, 0:4]
            round_keys[i, 8:12] = round_keys[i - 1, 8:12] ^ round_keys[i, 4:8]
            round_keys[i, 12:16] = round_keys[i - 1, 12:16] ^ round_keys[i, 8:12]
        return round_keys

    @staticmethod
    def shift_rows(state: np.ndarray):
        state[:] = state[shift_table]

    @staticmethod
    def shift_rows_inv(state: np.ndarray):
        state[:] = state[shift_table_inv]

    @staticmethod
    def mix_columns(state: np.ndarray):
        # for i in range(4):
        #     np.bitwise_xor.reduce(gf_mul(mix_matrix, state[i * 4:i * 4 + 4]), axis=1, out=state[i * 4:i * 4 + 4])
        state[:] = (mix_matrix_GF @ state.reshape(4, 4).T.view(GF)).T.ravel()

    @staticmethod
    def mix_columns_inv(state: np.ndarray):
        # for i in range(4):
        #     np.bitwise_xor.reduce(gf_mul(mix_matrix_inv, state[i * 4:i * 4 + 4]), axis=1, out=state[i * 4:i * 4 + 4])
        state[:] = (mix_matrix_inv_GF @ state.reshape(4, 4).T.view(GF)).T.ravel()

    def encrypt(self, plaintext: bytes) -> bytes:
        state = np.frombuffer(bytearray(plaintext), dtype=np.uint8)

        MyAES2.add_round_key(state, self.round_keys[0])
        for i in range(1, 10):
            MyAES2.sub_bytes(state)
            MyAES2.shift_rows(state)
            MyAES2.mix_columns(state)
            MyAES2.add_round_key(state, self.round_keys[i])
        MyAES2.sub_bytes(state)
        MyAES2.shift_rows(state)
        MyAES2.add_round_key(state, self.round_keys[10])

        return state.tobytes()

    def decrypt(self, ciphertext: bytes) -> bytes:
        state = np.frombuffer(bytearray(ciphertext), dtype=np.uint8)

        MyAES2.add_round_key(state, self.round_keys[10])
        for i in range(9, 0, -1):
            MyAES2.shift_rows_inv(state)
            MyAES2.sub_bytes_inv(state)
            MyAES2.add_round_key(state, self.round_keys[i])
            MyAES2.mix_columns_inv(state)
        MyAES2.shift_rows_inv(state)
        MyAES2.sub_bytes_inv(state)
        MyAES2.add_round_key(state, self.round_keys[0])

        return state.tobytes()


if __name__ == '__main__':
    import timeit

    from Cryptodome.Cipher import AES

    key = '1234567890123456'.encode()
    plaintext = 'abcdefghijklmnop'.encode()

    my_aes2 = MyAES2(key)
    ciphertext = my_aes2.encrypt(plaintext)
    print(ciphertext.hex())
    decrypted_plaintext = my_aes2.decrypt(ciphertext)
    print(decrypted_plaintext.decode())

    aes = AES.new(key, AES.MODE_ECB)
    ciphertext = aes.encrypt(plaintext)
    print(ciphertext.hex())
    decrypted_plaintext = aes.decrypt(ciphertext)
    print(decrypted_plaintext.decode())

    print(timeit.timeit(lambda: my_aes2.encrypt(plaintext), number=100))
    print(timeit.timeit(lambda: aes.encrypt(plaintext), number=100))
