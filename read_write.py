import csv


def read_column_from_csv(file_path, column_name):
    with open(file_path, mode="r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        image_names = []
        if column_name not in reader.fieldnames:
            print(f"Column '{column_name}' not found in the CSV file.")
            return

        for row in reader:
            image_names.append(row[column_name])
        return image_names


def add_column_to_csv(file_path, new_column_name, new_column_values):
    # Read the existing data from the CSV file
    with open(file_path, mode="r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames
        rows = list(reader)

    # Add the new column to each existing row
    for i, row in enumerate(rows):
        if i < len(new_column_values):
            row[new_column_name] = new_column_values[i]
        else:
            row[new_column_name] = (
                ""  # Add an empty value if there are more rows than new values
            )

    # Add any additional rows needed for extra new values
    for j in range(len(rows), len(new_column_values)):
        new_row = {name: "" for name in fieldnames}  # Create an empty row
        new_row[new_column_name] = new_column_values[j]
        rows.append(new_row)

    # Add the new column name to the list of fieldnames
    if new_column_name not in fieldnames:
        fieldnames.append(new_column_name)

    # Write the updated data back to the CSV file
    with open(file_path, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
