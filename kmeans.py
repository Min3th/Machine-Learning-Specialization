def closest_number_index(target_number, numbers_list):
    differences = [abs(target_number - num) for num in numbers_list]
    min_difference = min(differences)
    closest_number_index = differences.index(min_difference)
    return closest_number_index

# Example usage:
target = 10
numbers = [5, 8, 12, 15, 18]
index = closest_number_index(target, numbers)
print(f"The index of the number with the smallest difference is: {index}")
