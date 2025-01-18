import json
import os
from pathlib import Path


def process_exercise_files(directory_path):
    with open('exercise_summary.txt', 'w') as outfile:
        # Write the header
        outfile.write("Exercise Name | Primary Muscles | Secondary Muscles | Equipment | Category\n")
        # Iterate through all JSON files in the directory
        for filename in Path(directory_path).glob('*.json'):
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)

                # Extract the required fields
                exercise_name = data.get('name', '')
                primary_muscles = ','.join(data.get('primaryMuscles', []))
                secondary_muscles = ','.join(data.get('secondaryMuscles', []))
                equipment = data.get('equipment', '')
                category = data.get('category', '')

                # Create a summary line
                summary_line = f"{exercise_name} | {primary_muscles} | {secondary_muscles} | {equipment} | {category}\n"

                # Write to output file
                outfile.write(summary_line)

            except json.JSONDecodeError:
                print(f"Error reading JSON file: {filename}")
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")


if __name__ == "__main__":
    directory_path = "/Users/mikhail/PycharmProjects/healthshape-works/exercises"
    process_exercise_files(directory_path)
