# from functools import cache

import numpy as np
import galois

from aes_tables import RCONs, S_box, mix_matrix, mix_matrix_inv, S_box_inv, GF_poly

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


GF = galois.GF(2 ** 8, irreducible_poly=GF_poly)
mix_matrix_GF = mix_matrix.view(GF)
mix_matrix_inv_GF = mix_matrix_inv.view(GF)


class MyAES:
    def __init__(self, key: bytes):
        self.key = key
        self.round_keys = MyAES.get_round_keys(key)

    @staticmethod
    def sub_bytes(state: np.ndarray):
        state[:] = S_box[state]

    @staticmethod
    def sub_bytes_inv(state: np.ndarray):
        state[:] = S_box_inv[state]

    @staticmethod
    def g(word: np.ndarray, RCON: np.uint8) -> np.ndarray:
        word = np.roll(word, -1)
        MyAES.sub_bytes(word)
        word[0] ^= RCON
        return word

    @staticmethod
    def get_round_keys(key: bytes) -> np.ndarray:
        round_keys = np.empty((11, 4, 4), dtype=np.uint8)
        round_keys[0] = np.frombuffer(key, dtype=np.uint8).reshape(4, 4).T
        for i in range(1, 11):
            round_keys[i, :, 0] = round_keys[i - 1, :, 0] ^ MyAES.g(round_keys[i - 1, :, 3], RCONs[i - 1])
            round_keys[i, :, 1] = round_keys[i - 1, :, 1] ^ round_keys[i, :, 0]
            round_keys[i, :, 2] = round_keys[i - 1, :, 2] ^ round_keys[i, :, 1]
            round_keys[i, :, 3] = round_keys[i - 1, :, 3] ^ round_keys[i, :, 2]
        return round_keys

    @staticmethod
    def add_round_key(state: np.ndarray, round_key: np.ndarray):
        state ^= round_key

    @staticmethod
    def shift_rows(state: np.ndarray):
        for i in range(1, 4):
            state[i] = np.roll(state[i], -i)

    @staticmethod
    def shift_rows_inv(state: np.ndarray):
        for i in range(1, 4):
            state[i] = np.roll(state[i], i)

    @staticmethod
    def mix_columns(state: np.ndarray):
        # for i in range(4):
        #     np.bitwise_xor.reduce(gf_mul(mix_matrix, state[:, i]), axis=1, out=state[:, i])
        state[:] = mix_matrix_GF @ state.view(GF)

    @staticmethod
    def mix_columns_inv(state: np.ndarray):
        # for i in range(4):
        #     np.bitwise_xor.reduce(gf_mul(mix_matrix_inv, state[:, i]), axis=1, out=state[:, i])
        state[:] = mix_matrix_inv_GF @ state.view(GF)

    def encrypt(self, plaintext: bytes) -> bytes:
        state = np.frombuffer(bytearray(plaintext), dtype=np.uint8).reshape(4, 4).T

        MyAES.add_round_key(state, self.round_keys[0])
        for i in range(1, 10):
            MyAES.sub_bytes(state)
            MyAES.shift_rows(state)
            MyAES.mix_columns(state)
            MyAES.add_round_key(state, self.round_keys[i])
        MyAES.sub_bytes(state)
        MyAES.shift_rows(state)
        MyAES.add_round_key(state, self.round_keys[10])

        return state.T.tobytes()

    def decrypt(self, ciphertext: bytes) -> bytes:
        state = np.frombuffer(bytearray(ciphertext), dtype=np.uint8).reshape(4, 4).T

        MyAES.add_round_key(state, self.round_keys[10])
        for i in range(9, 0, -1):
            MyAES.shift_rows_inv(state)
            MyAES.sub_bytes_inv(state)
            MyAES.add_round_key(state, self.round_keys[i])
            MyAES.mix_columns_inv(state)
        MyAES.shift_rows_inv(state)
        MyAES.sub_bytes_inv(state)
        MyAES.add_round_key(state, self.round_keys[0])

        return state.T.tobytes()


if __name__ == '__main__':
    import timeit

    from Cryptodome.Cipher import AES

    key = '1234567890123456'.encode()
    plaintext = 'abcdefghijklmnop'.encode()

    my_aes = MyAES(key)
    ciphertext = my_aes.encrypt(plaintext)
    print(ciphertext.hex())
    decrypted_plaintext = my_aes.decrypt(ciphertext)
    print(decrypted_plaintext.decode())

    aes = AES.new(key, AES.MODE_ECB)
    ciphertext = aes.encrypt(plaintext)
    print(ciphertext.hex())
    decrypted_plaintext = aes.decrypt(ciphertext)
    print(decrypted_plaintext.decode())

    print(timeit.timeit(lambda: my_aes.encrypt(plaintext), number=100))
    print(timeit.timeit(lambda: aes.encrypt(plaintext), number=100))
