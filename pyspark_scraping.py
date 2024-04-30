from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
import re

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("TextProcessing") \
    .getOrCreate()

# Define UDF for removing extra whitespace
def remove_extra_whitespace(text):
    return re.sub("\n+", "\n", text)

remove_extra_whitespace_udf = udf(remove_extra_whitespace, StringType())

# Define UDF for checking if a line is likely part of a figure
def is_figure_line(line, threshold=0.65):
    non_alpha_num_chars = sum(not char.isalnum() for char in line)
    total_chars = len(line)
    if total_chars == 0:
        return False
    density = non_alpha_num_chars / total_chars
    return density > threshold

is_figure_line_udf = udf(is_figure_line)

# Define UDF for extracting text
def extract_text(line, figure_start_end_markers=('FIGURE START', 'FIGURE END')):
    line = line.strip()
    if line.startswith(figure_start_end_markers[0]) or line.startswith(figure_start_end_markers[1]):
        return ''
    else:
        return line

extract_text_udf = udf(extract_text, StringType())

# Define page_content function
def page_content(text):
    pattern1 = r'^[A-Za-z]+\s+\[Page \d+\]$'
    pattern2 = r'^RFC \d+\s+.+$'

    result = re.sub(pattern1, '', text, flags=re.MULTILINE)
    result = re.sub(pattern2, '', result, flags=re.MULTILINE)
    
    return result

page_content_udf = udf(page_content, StringType())

# Read input file into DataFrame
input_file_path = "file1.txt"
df = spark.read.text(input_file_path)

# Apply page_content function
df = df.withColumn("modified_text", 
                   page_content_udf("value"))

# Apply extract_text function
figure_start_end_markers = ('FIGURE START', 'FIGURE END')
df = df.withColumn("modified_text", extract_text_udf("modified_text", figure_start_end_markers))

# Apply remove_extra_whitespace function
df = df.withColumn("modified_text", remove_extra_whitespace_udf("modified_text"))

# Save result
output_file_path = "output1.txt"
df.select("modified_text").write.mode("overwrite").text(output_file_path)

# Stop SparkSession
spark.stop()
