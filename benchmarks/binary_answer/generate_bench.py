import os
import sys
import csv
from openai import OpenAI

def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def get_csv_reader(file_path):
    # We open the file here, but note that in the main loop we need to 
    # ensure we handle file closing or re-opening correctly.
    file = open(file_path, 'r', newline='', encoding='utf-8')
    reader = csv.reader(file)
    return reader, file

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python generate_bench.py [openai_model_name] [output_folder]")
        print("Example: python generate_bench.py gpt-5.2 ./benchmarks/binary/answer/gen_v1")
        sys.exit(1)
        
    model_id, output_folder = sys.argv[1], sys.argv[2]

    # Initialize OpenAI Client
    api_key = os.getenv("OPENAI_API_KEY")
        
    client = OpenAI(api_key=api_key)

    # Prepare output directory
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)

    prompt_true = load_text_file("./benchmarks/binary_answer/prompt_true.txt")
    prompt_false = load_text_file("./benchmarks/binary_answer/prompt_false.txt")

    output_path = os.path.join(output_folder, "bench.csv")
    
    # Check if we need to write headers (if file doesn't exist)
    # write_headers = not os.path.exists(output_path)

    with open(output_path, 'a', newline='', encoding='utf-8') as outputs:
        writer = csv.writer(outputs)
        # if write_headers:
        writer.writerow(["characteristic", "question", "answer"])

        for answer in ['yes', 'no']:
            # Re-open the input CSV for every loop iteration to reset the reader
            reader, input_file = get_csv_reader("./data/characteristics.txt")
            
            content_template = prompt_true if answer == 'yes' else prompt_false
            
            for n, row in enumerate(reader):
                if not row: continue # Skip empty lines

                messages = [
                    {"role": "user", "content": content_template.format(characteristic=row[0])},
                ]

                try:
                    response = client.chat.completions.create(
                        model=model_id,
                        messages=messages,
                        temperature=0.5,
                        top_p=0.9,
                        # OpenAI uses frequency_penalty (-2.0 to 2.0) instead of repetition_penalty (>1.0)
                        # Leaving as default 0 or slight positive for variety
                        frequency_penalty=0.0 
                    )
                    generated_text = response.choices[0].message.content

                    writer.writerow([row[0], generated_text, answer])
                    print(f"Completed {n+1} {answer} text ideas: \n{generated_text}\n -------\n")
                
                except Exception as e:
                    print(f"Error processing row {n}: {e}")

            # Close input file before next iteration
            input_file.close()