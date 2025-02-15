import operator
from functools import reduce

from aes_2d import MyAES
from aes_tables import shift_table, shift_table_inv, mix_matrix, mix_matrix_inv


def gf_mul(a: int, b: int) -> int:
    p = 0
    while b:
        if b & 1: p ^= a
        msb = a & 0x80
        a <<= 1
        if msb: a ^= 0x1b
        b >>= 1
    return p & 0xff


class MyAES2(MyAES):
    def __init__(self, key: bytes):
        super().__init__(key)

    @staticmethod
    def add_round_key(state: list[int], key: bytes | list[int]) -> list[int]:
        return list(map(operator.xor, state, key))

    @staticmethod
    def sub_bytes(state: list[int]) -> list[int]:
        return list(map(MyAES.S, state))

    @staticmethod
    def sub_bytes_inv(state: list[int]) -> list[int]:
        return list(map(MyAES.S_inv, state))

    @staticmethod
    def shift_rows(state: list[int] | bytes) -> list[int]:
        return list(map(getattr(state, '__getitem__'), shift_table))

    @staticmethod
    def shift_rows_inv(state: list[int]) -> list[int]:
        return list(map(getattr(state, '__getitem__'), shift_table_inv))

    @staticmethod
    def mix_columns(state: list[int]) -> list[int]:
        return [reduce(operator.xor, map(gf_mul, mix_matrix[j], state[i * 4:i * 4 + 4])) for i in range(4) for j in range(4)]

    @staticmethod
    def mix_columns_inv(state: list[int]) -> list[int]:
        return [reduce(operator.xor, map(gf_mul, mix_matrix_inv[j], state[i * 4:i * 4 + 4])) for i in range(4) for j in range(4)]

    def encrypt(self, plaintext: bytes) -> bytes:
        state = list(plaintext)

        state = self.add_round_key(state, self.round_keys[0])
        for i in range(1, 10):
            state = self.sub_bytes(state)
            state = self.shift_rows(state)
            state = self.mix_columns(state)
            state = self.add_round_key(state, self.round_keys[i])
        state = self.sub_bytes(state)
        state = self.shift_rows(state)
        state = self.add_round_key(state, self.round_keys[10])

        return bytes(state)

    def decrypt(self, ciphertext: bytes) -> bytes:
        state = list(ciphertext)

        state = self.add_round_key(state, self.round_keys[10])
        for i in range(9, 0, -1):
            state = self.shift_rows_inv(state)
            state = self.sub_bytes_inv(state)
            state = self.add_round_key(state, self.round_keys[i])
            state = self.mix_columns_inv(state)
        state = self.shift_rows_inv(state)
        state = self.sub_bytes_inv(state)
        state = self.add_round_key(state, self.round_keys[0])

        return bytes(state)


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
