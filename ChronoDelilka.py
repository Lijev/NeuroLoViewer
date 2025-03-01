def split_video():
    while True:
        try:
            duration = input("\nEnter the timing of the video (in minutes with seconds after the decimal point, or 'exit' to exit): ")
            if duration.lower() == 'exit':
                print(" Quitting...")
                break  

            duration = float(duration)  
            duration_seconds = duration * 60  
            segment_length = duration_seconds / 10  

            print("\nSplitting video in 10 chunks")
            for i in range(1, 11):
                segment_time = segment_length * i
                minutes = int(segment_time // 60)
                seconds = round(segment_time % 60, 2)
                print(f"chunk {i}: {minutes} min {seconds} sec")

        except ValueError:
            print("Error: Enter a valid number (e.g. 5.30 for 5 minutes 30 seconds) or 'exit' to exit.")

split_video()
