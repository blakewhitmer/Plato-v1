D = 73584077
N = D / 20
print( N )

C = 120 * N ** 2
print(f"{C:.2e} floating point operations")
fp_per_hour = 11.36 * 10 ** 12 * 60 * 60
print(C / (fp_per_hour / 24))

# Model has 3448064 parameters