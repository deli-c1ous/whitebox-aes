import operator
from functools import reduce, cache

from more_itertools import sliced, batched

from aes_tables import RCONs, S_box, mix_matrix, mix_matrix_inv


@cache
def gf_mul(a: int, b: int) -> int:
    p = 0
    while b:
        if b & 1: p ^= a
        msb = a & 0x80
        a <<= 1
        if msb: a ^= 0x1b
        b >>= 1
    return p & 0xff


def left_rotate(a: bytes | list, n) -> bytes | list:
    return a[n:] + a[:n]


def bytes_xor(a: bytes, b: bytes) -> bytes:
    return bytes(map(operator.xor, a, b))


def bytes2matrix(data: bytes) -> list[list[int]]:
    return list(map(lambda x: list(x), zip(*batched(data, 4))))


def matrix2bytes(matrix: list[list[int]]) -> bytes:
    return b''.join(map(lambda x: bytes(x), zip(*matrix)))


class MyAES:
    def __init__(self, key: bytes):
        self.key = key
        self.round_keys = MyAES.get_round_keys(key)

    @staticmethod
    def S(byte: int) -> int:
        return S_box[byte]

    @staticmethod
    def S_inv(byte: int) -> int:
        return S_box.index(byte)

    @staticmethod
    def g(word: bytes, RCON: bytes) -> bytes:
        word = left_rotate(word, 1)
        word = bytes(map(MyAES.S, word))
        word = bytes_xor(word, RCON)
        return word

    @staticmethod
    def get_round_keys(key: bytes) -> list[bytes]:
        new_RCONs = list(map(lambda x: x.to_bytes(4, 'little'), RCONs))
        words = list(sliced(key, 4))
        for i in range(4, 44):
            tmp = words[i - 1]
            if i % 4 == 0: tmp = MyAES.g(tmp, new_RCONs[i // 4 - 1])
            words.append(bytes_xor(words[i - 4], tmp))
        round_keys = batched(words, 4)
        round_keys = list(map(b''.join, round_keys))
        return round_keys

    @staticmethod
    def add_round_key(state: list[list[int]], round_key: bytes) -> list[list[int]]:
        round_key_matrix = bytes2matrix(round_key)
        return list(map(lambda x, y: list(map(operator.xor, x, y)), state, round_key_matrix))

    @staticmethod
    def sub_bytes(state: list[list[int]]) -> list[list[int]]:
        return list(map(lambda x: list(map(MyAES.S, x)), state))

    @staticmethod
    def sub_bytes_inv(state: list[list[int]]) -> list[list[int]]:
        return list(map(lambda x: list(map(MyAES.S_inv, x)), state))

    @staticmethod
    def shift_rows(state: list[list[int]]) -> list[list[int]]:
        return list(map(lambda i: left_rotate(state[i], i), range(4)))

    @staticmethod
    def shift_rows_inv(state: list[list[int]]) -> list[list[int]]:
        return list(map(lambda i: left_rotate(state[i], 4 - i), range(4)))

    @staticmethod
    def mix_columns(state: list[list[int]]) -> list[list[int]]:
        return list(map(list, zip(*map(lambda x: map(lambda y: reduce(operator.xor, map(gf_mul, x, y)), mix_matrix), zip(*state)))))

    @staticmethod
    def mix_columns_inv(state: list[list[int]]) -> list[list[int]]:
        return list(map(list, zip(*map(lambda x: map(lambda y: reduce(operator.xor, map(gf_mul, x, y)), mix_matrix_inv), zip(*state)))))

    def encrypt(self, plaintext: bytes) -> bytes:
        state = bytes2matrix(plaintext)

        state = MyAES.add_round_key(state, self.round_keys[0])
        for i in range(1, 10):
            state = MyAES.sub_bytes(state)
            state = MyAES.shift_rows(state)
            state = MyAES.mix_columns(state)
            state = MyAES.add_round_key(state, self.round_keys[i])
        state = MyAES.sub_bytes(state)
        state = MyAES.shift_rows(state)
        state = MyAES.add_round_key(state, self.round_keys[10])

        ciphertext = matrix2bytes(state)
        return ciphertext

    def decrypt(self, ciphertext: bytes) -> bytes:
        state = bytes2matrix(ciphertext)

        state = MyAES.add_round_key(state, self.round_keys[10])
        for i in range(9, 0, -1):
            state = MyAES.shift_rows_inv(state)
            state = MyAES.sub_bytes_inv(state)
            state = MyAES.add_round_key(state, self.round_keys[i])
            state = MyAES.mix_columns_inv(state)
        state = MyAES.shift_rows_inv(state)
        state = MyAES.sub_bytes_inv(state)
        state = MyAES.add_round_key(state, self.round_keys[0])

        plaintext = matrix2bytes(state)
        return plaintext


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
