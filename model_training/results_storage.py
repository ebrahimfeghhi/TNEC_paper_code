import os 
import errno 

def results_storage(run_number):
    

    results_storage_path = '/home3/ebrahim/results/test_detector/cnn/final_models/' + run_number
        
    try:
        os.makedirs(results_storage_path, exist_ok=True)
        print("Folder created")
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    
    return results_storage_path