OTB2013Set = {'Basketball', 'Doll', 'Bolt', 'Soccer', 'Deer', 'Boy', 'CarDark', 'CarScale',...
          'David', 'David3', 'Football1', 'Girl', 'Crossing',...
          'MountainBike', 'Shaking', 'Singer1', 'Singer2', 'Skating1',...
          'Trellis', 'Walking', 'Walking2', 'Woman', 'Tiger1', 'MotorRolling',...
          'Lemming', 'Matrix', 'Coke', 'FaceOcc1', 'Liquor', 'Skiing', 'Tiger2', 'Ironman', 'Couple',...
          'Car4', 'Subway', 'David2', 'Dog1', 'Dudek', 'FaceOcc2', 'Fish', 'FleetFace',...
          'Football', 'Freeman1', 'Freeman3', 'Freeman4', 'Jogging-1', 'Jogging-2',...
          'Jumping', 'Mhyang', 'Suv', 'Sylvester'};

base_path = './sequences/';
res_path = './res/results_';
testSet = OTB2013Set;  
addpath('res');

fid_rec = fopen('record.txt', 'wt');
 
    average_center_location_error_sum=0;
    distance_precision_sum=0;
    PASCAL_precision_sum=0;
    Overlap_sum=0;
    fps_sum=0;
    for i = 1:length(testSet)
        video_path = [base_path testSet{i} '/'];
        [img_files, pos, target_sz, ground_truth, video_path] = ...
            load_video_info(video_path, testSet{i});
        result_path = [res_path  testSet{i} '.txt'];
        fid = fopen(result_path, 'rt');
        res = textscan(fid, '%d,%d,%d,%d');
        positions = [res{1},res{2},res{3},res{4}];
        positions = single(positions);
        [distance_precision, PASCAL_precision, average_center_location_error, Overlap] = ...
            compute_performance_measures(positions, ground_truth, 20, 0.5);
       disp([testSet{i} ': ' num2str(Overlap)]);
        average_center_location_error_sum=average_center_location_error_sum+average_center_location_error;
        distance_precision_sum=distance_precision_sum+distance_precision;
        PASCAL_precision_sum=PASCAL_precision_sum+PASCAL_precision;
        Overlap_sum=Overlap_sum+Overlap;
        fclose(fid);
    end
    average_center_location_error=average_center_location_error_sum/length(testSet);
    distance_precision=distance_precision_sum/length(testSet);
    PASCAL_precision=PASCAL_precision_sum/length(testSet);
    Overlap=Overlap_sum/length(testSet);

    fprintf('Center Location Error: %.3g pixels\nDistance Precision: %.3g %%\nOverlap Precision: %.5g %%\nOverlap: %.5g%%\n', ...
        average_center_location_error, 100*distance_precision, 100*PASCAL_precision,100*Overlap);
    
fclose(fid_rec);