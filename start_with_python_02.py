# 제어문

# def print_if_positive(number):
#     if number >= 0:
#         print(number)

# 숫자가 0보다 크거나 같을(>=) 경우, 즉 0 이상일 경우에만 숫자를 출력

# print_if_positive(1)
# print_if_positive(-1)

# def print_if_positive(number):
#     if number >= 0:
#         print('+')
#     else:
#         print('-')

# 숫자가 0보다 크거나 같으면 +를, 작으면 -를 출력
# print_if_positive(1)
# print_if_positive(-1)

# def print_if_positive(number):
#     if number > 0:
#         print("+")
#     elif number ==0:
#         print(0)
#     else:
#         print('-')

# 숫자가 양수이면 +, 0이면 0, 음수이면 -
# print_if_positive(1)
# print_if_positive(0)
# print_if_positive(-1)


# def print_if_positive_and_even(number):
#     if (number > 0) and (number % 2==0):
#         print(number)

        # 숫자가 양수이고 짝수일 때 해당 숫자를 출력
# print_if_positive_and_even(1)
# print_if_positive_and_even(-1)
# print_if_positive_and_even(2)
# print_if_positive_and_even(-2)

# def print_if_negative_or_odd(number):
#     if (number <0) or (number%2!=0):
#         print(number)

        # 숫자가 음수이거나 홀수 일 때 해당 숫자 출력

# print_if_negative_or_odd(1)
# print_if_negative_or_odd(-1)
# print_if_negative_or_odd(2)
# print_if_negative_or_odd(-2)


# def print_if_negative_or_odd(number):
#     if (number<=0) or (number % 2 != 0):
#         print(number)

        # [5-8] print_if_negative_or_odd 함수

# print_if_negative_or_odd(1)
# print_if_negative_or_odd(-1)
# print_if_negative_or_odd(2)
# print_if_negative_or_odd(-2)

# 피보나치

# def fibonacci(n):
#     if n <=2:
#         return 1
#     else:
#         return fibonacci(n-2) + fibonacci(n-1)

# n = 1

# while n <20:
#     print(fibonacci(n))
#     n = n+1

# print('끝!')



# 3-12. for

# for character in 'Hello':
#     print(character)

