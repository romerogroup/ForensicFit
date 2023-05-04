from data.data_ingestion import DataIngestion
from data.data_generation import DataGenerator

def main():
    data_ingestion = False
    generate_data = True


    if data_ingestion:
        obj=DataIngestion(from_scratch=False)
        obj.initiate_data_ingestion()
        
    if generate_data:
        obj=DataGenerator(from_scratch=False)
        # obj.initiate_normal_split_generation(generated_dir='match_nonmatch_ratio_0.3', match_nonmatch_ratio=0.3)
        obj.initiate_cross_validation_generation(generated_dir='match_nonmatch_ratio_0.3', match_nonmatch_ratio=0.3)



if __name__ == '__main__':
    main()