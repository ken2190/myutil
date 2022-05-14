from pyspark.sql.functions import regexp_replace, asc

input_df = spark.read.format("binaryFile").option("pathGlobFilter", "*{}".format(FORMAT)).load(INPUT_DIR_DBFS).withColumn("path", regexp_replace("path", "dbfs:/", "/dbfs/")).select("path", "modificationTime", "length")
display(input_df.orderBy(asc("path")))



from pyspark.sql.functions import udf, col, lit, explode

@udf("array<string>")
def split_video_files(video_file, modification_time, split_by_seconds, output_dir):
  
  import os
  from os.path import basename
  from moviepy.editor import VideoFileClip
  from datetime import timedelta 
  from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
  
  clip = VideoFileClip(video_file)
  duration = round(clip.duration + 0.5) # Use 0.5 so we always round up, as opposed to down...

  extracted_list = []  
  for i in range(0, duration, split_by_seconds):   
    
    start, end = i, i + split_by_seconds
    time_format = "%Y-%m-%dT%H:%M:%S"
    start_time, end_time = modification_time + timedelta(seconds=start), modification_time + timedelta(seconds=end)
    output_filename = f"start_{start_time.strftime(time_format)}_end_{end_time.strftime(time_format)}_{basename(video_file)}"
    local_file = f"/tmp/{output_filename}"
    
    if not os.path.isfile(output_filename):
      # Check whether the target file exists or not before creating it
      ffmpeg_extract_subclip(video_file, start, end, local_file) 
      output_path = output_dir + output_filename
    
    try:
      import shutil
      # copy-then-delete, because it's moving between different filesystems, 
      # and writing locally before copying to an NFS mount is safer
      shutil.move(local_file, output_path)
      extracted_list.append(output_path)
    except Exception as e:
      print(f"Error moving {local_file} to {video_file} : {e}")
  return extracted_list

output_df = input_df.withColumn("output_file", explode(split_video_files(col("path"), col("modificationTime"), lit(SPLIT_BY_SECONDS), lit(SPLIT_DIR)))).drop("length")
display(output_df)