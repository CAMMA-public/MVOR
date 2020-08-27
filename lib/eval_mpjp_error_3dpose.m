%%
%File: eval_mpjp_error_3dpose.m
%Project: CAMMA-MVOR
%-----
%Copyright (c) University of Strasbourg, All Rights Reserved.
%%

function eval_mpjp_error_3dpose( result_file )
% Read the mv3dreg output stored in a mat file and shows the result
ll= load(result_file);
[errorPerJointPerSupport, errorJoints, nSupportSkel] = errorReport(ll.enc_in, ll.GT3dPoses, ll.det3dPoses);
avgError(1, :) = mean(errorJoints);
r = errorPerJointPerSupport;
r1 = int64(r);
one_view = [(r(1, 3)+r(1, 5))/20.0  (r(1, 4)+r(1, 6))/20.0 (r(1, 7)+r(1, 8))/20.0 r(1, 11)/10.0];
two_view = [(r(2, 3)+r(2, 5))/20.0  (r(2, 4)+r(2, 6))/20.0 (r(2, 7)+r(2, 8))/20.0 r(2, 11)/10.0];
three_view = [(r(3, 3)+r(3, 5))/20.0  (r(3, 4)+r(3, 6))/20.0 (r(3, 7)+r(3, 8))/20.0 r(3, 11)/10.0];
mean_one_view = round(mean(one_view),1);
mean_two_view = round(mean(two_view), 1);
mean_three_view = round(mean(three_view), 1);
one_view = round(one_view, 1);
two_view = round(two_view, 1);
three_view = round(three_view,1);
avgError_perexperiment = mean(avgError(:, [3,4,5,6,7,8,11]),2);
%disp('results per joint')
%disp(errorPerJointPerSupport)
disp('--       Shoulder      Elbow       Wrist         Hip     Average --')
disp(['oneview   ' num2str(one_view) '     ' num2str(mean_one_view)])
disp(['twoview   ' num2str(two_view) '     ' num2str(mean_two_view)])
disp(['threeview ' num2str(three_view) '     ' num2str(mean_three_view)])


    function [errorPerJointPerSupport, errorJoints, nSupportSkel] = errorReport(in2d, gt3d, out3d)
        
        H36M2MCT = [15 14 25 26 17 18 27 19 1 6] +1;
        H36M2MCT = H36M2MCT(3:end); %we remove head and neck
        nSamples = size(in2d,1);
        assert(size(gt3d,1) == nSamples && size(out3d,1) == nSamples);
        nViews = size(in2d,2) / (32*3);
        assert(nViews * 32* 3 == size(in2d,2));
        
        scores2d = in2d(:,3:3:end);
        nDetectedPartPerview = zeros(nSamples, nViews);
        for vidx = 1 : nViews
            vOffset = (vidx -1) * 32;
            tmp = scores2d(:, vOffset + H36M2MCT );
            tmp = tmp > 0.001;
            nDetectedPartPerview(:, vidx) = sum(tmp,2);
        end
        
        nSupportSkel = sum(nDetectedPartPerview > 2,2);
        
        % add hip center
        out3d(:, end+1:end+3) = (out3d(:, 25:27) + out3d(:, 28:30) ) ./2.0;
        gt3d(:, end+1:end+3) = (gt3d(:, 25:27) + gt3d(:, 28:30) ) ./2.0;
        
        errorJoints = nan(nSamples, size(out3d,2)/3);
        for i =1 : 11
            pcoordinate = (i-1)*3+1 : i*3;
            errorJoints(:,i) = sqrt(sum((gt3d(:,pcoordinate)-out3d(:,pcoordinate)).^2,2));
        end
        
        errorPerJointPerSupport = zeros(nViews+1, size(out3d,2)/3);
        for i = 1 : nViews
            skelIds = nSupportSkel==i;
            errorPerJointPerSupport(i, :) = mean(errorJoints(skelIds, :));
        end
        errorPerJointPerSupport(end, :) = mean(errorJoints(1:end-1, :));
    end
end



