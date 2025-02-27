def split_video():
    try:
        duration = float(input("Enter the video timing (in minutes with seconds after the decimal point): "))
        duration_seconds = duration * 60  
        segment_length = duration_seconds / 10  

        print("\nDividing the video into 10 equal chunks:")
        for i in range(1, 11):
            segment_time = segment_length * i
            minutes = int(segment_time // 60)
            seconds = round(segment_time % 60, 2)
            print(f"chunk {i}: {minutes} min {seconds} sec")

    except ValueError:
        print("Error: input correct valueble (exmp, 5.30 for 5 minutes 30 seconds).")

split_video()
