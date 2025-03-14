%This is the original post processing script created by Yu shi with the functions updated to match the latest function signatures in the latest git release of Petakit 5-D
%Present in : S:\Yu\PetaKit5D-main\LegantLab_Script\ on stegasaurus

%%  Open LLSM5D_tool folder and setup environment
% Note: please make sure your GPU has at least 32 GB available vRAM for
% deconvolution on GPU to avoid out-of-memory issue; otherwise, you may run CPU deconvolution.

clear, clc;

% fprintf('Large-scale processing demo...\n\n');

% move to the LLSM5DTools root directory
curPath = pwd;
if ~endsWith(curPath, 'LLSM5DTools')
    mfilePath = mfilename('fullpath');
    if contains(mfilePath,'LiveEditorEvaluationHelper')
        mfilePath = matlab.desktop.editor.getActiveFilename;
    end
    
    mPath = fileparts(mfilePath);
    if endsWith(mPath, 'LegantLab_Script')
        cd(mPath);
        cd('..')
    end
end

setup();

%% Setting up parameters
% root path
rt = 'X:\WRL\2025_01_28(SmartMicroMitosis3D)'; %'S:\VGA\2024_03_08_LLSM_ExM';
% data path for data to be deconvolved, also support for multiple data folders
dataPaths = {[rt, '\3D_array\']};  %\test_run_medium_512pxl_25um\

% xy pixel size in um
xyPixelSize = 0.110002 ; %0.108;
% z step size
dz = 0.5;
% scan direction
Reverse = true;
% psf z step size (we assume xyPixelSize also apply to psf)
dzPSF = 0.5;

% skew angle
SkewAngle = 32.45;

% resolution [xyPixelsize, dz]
Resolution = [xyPixelSize, dz];

% if true, check whether image is flipped in z using the setting files
parseSettingFile = false;

% channel patterns for the channels, the channel patterns should map the
% order of PSF filenames.
%ChannelPatterns = {'CamA_ch0', 'CamB_ch1', 'CamB_ch2'};  
ChannelPatterns = {'CamA_ch0', 'CamB_ch0', 'CamB_ch1', 'CamB_ch2'};  
% ChannelPatterns = {'CamA_ch0'};  

% psf path
psf_rt = rt;            
PSFFullpaths = {
                [psf_rt, '\Calibration\skewXstagePSF_488.tif'], ...
                [psf_rt, '\Calibration\skewXstagePSF_560.tif'], ...
                [psf_rt, '\Calibration\skewXstagePSF_642.tif'], ...
                };           
% PSFFullpaths = {
%                 [psf_rt, '\Calibration\skewXstagePSF_488.tif'], ...
%                 };    
%% Generate ImageList from encoder
% generate positions for a list of slices, file name is "ImageList_from_encoder.csv"
stitch_generate_imagelist_from_encoder(dataPaths{1}, dz, ChannelPatterns);
% stitch_generate_imagelist_from_sqlite(dataPaths{1});  % uncomment this if you want to generate from sqlite database

%% stitch specific parameters
% image list path: csv file
% if not available, run stitch_generate_imagelist_from_encoder(dataPath, dz)
ImageListFullpath = [dataPaths{1}, '\ImageList_from_encoder.csv'];
% ImageListFullpath = [dataPaths{1}, '\ImageList_from_sqlite.csv'];

% stitch in DS space
DS = false;
% stitch in Deskew/rotated space, if both DS and DSR false, stitch in skewed space
DSR = false;

% axis order
axisOrder = '-x,y,z';

% stitch blending method, 'none': no blending, 'feather': feather blending
BlendMethod = 'feather';

% stitch dir string, inside dataPath
stitchResultDir = 'matlab_stitch';

% cross correlation registration, if false, directly stitch by the coordinates.
xcorrShift = true;

%if true, only stitch first time point
onlyFirstTP = false;

% stitch pipeline, no need to change, zarr pipeline is the mostly used one.
stitchPipeline = 'zarr';

% xcorr registration on the primary channel, and other channels uses the
% registration informaion from the primary channel
PrimaryCh = 'CamB_ch0';%'CamB_ch1';

% if true, save result as 16bit, if false, save as single
Save16bit = true;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% parameters below are used in special cases or for advanced fine tuning

% resample data, if empty, stitch in original resolution. If a 1X3 array,
% resample data and then stitch, mostly used for initial inspection
resample = [];

% resample type: isotropic, xy_isotropic; effective when 'resample' is
% not set. Keep it as 'isotropic'. 
resampleType = 'isotropic';

% max allowed shift (in pixel) in xy axes between neighboring tiles in xcorr registration. 
xyMaxOffset = 150;

% max allowed shift in z (in pixel) axis between neighboring tiles in xcorr registration. 
zMaxOffset = 150;

% downsampling factors for overlap regions to calculate xcorr in xcorr registration. 
% with larger downsamplinf factors, the faster of the xcorr computing, yet
% lower accuracy will be. 
xcorrDownsample = [2, 2, 2];

% crop input tile before any processing, empty (no crop) or a 1X6 array [ymin, xmin, zmin, ymax, xmax, zmax]
InputBbox = [];

% crop input tile after processing, empty (no crop) or a 1X6 array [ymin, xmin, zmin, ymax, xmax, zmax]
tileOutBbox = [];

% chunk size in zarr
blockSize = [256, 256, 256];

% Currently, flat field correction is disabled in the demo. Comment this line to enable it.
processFunPath = '';

% tile offset: counts add to the image, used when the image background is
% 0, to differentiate between image background and empty space. 
TileOffset = 0;

% use slurm cluster if true, otherwise use the local machine (master job)
parseCluster = false;
% use master job for task computing or not. 
masterCompute = true;
% configuration file for job submission
configFile = '';
% if true, use Matlab runtime (for the situation without matlab license)
mccMode = false;

%% Stitch images
tic
XR_matlab_stitching_wrapper(dataPaths, ImageListFullpath, ...
    'DS', DS, 'DSR', DSR, 'SkewAngle', SkewAngle, 'Reverse', Reverse, 'axisOrder', axisOrder, ...
    'xcorrShift', xcorrShift, 'ChannelPatterns', ChannelPatterns, 'PrimaryCh', PrimaryCh, ...
    'blockSize', blockSize, 'TileOffset', TileOffset, 'resampleType', resampleType, ...
    'resampleFactor', resampleFactor, 'xyPixelSize', xyPixelSize,'dz',dz,'BlendMethod', BlendMethod, ...   %'resample',resample,'Resolution', Resolution,
    'resultDir', stitchResultDir,  'xyMaxOffset', xyMaxOffset, ...     %'onlyFirstTP', onlyFirstTP,
    'zMaxOffset', zMaxOffset, 'xcorrDownsample', xcorrDownsample, 'InputBbox', InputBbox, ...
    'tileOutBbox', tileOutBbox, 'parseSettingFile', parseSettingFile, ...  %'pipeline', stitchPipeline,  (latest ver.)'zarrFile',zarrFile,
    'processFunPath', processFunPath, 'Save16bit', Save16bit, 'parseCluster', parseCluster, ...
    'masterCompute', masterCompute, 'configFile', configFile, 'mccMode', mccMode);
toc


%% RL method
dataPaths_exp = [dataPaths{1}, 'matlab_stitch\'];

RLmethod = 'simplified';
% deconvolution result path string (within dataPath)
deconPathstr = 'matlab_decon'; %matlab_decon_conventional
 
% background to subtract
Background = 100;
% number of iterations
DeconIter = 10;
% decon to 80 iterations (not use the criteria for early stop)
fixIter = true;
% erode the edge after decon for number of pixels.
EdgeErosion = 8;
% save as 16bit, if false, save to single
Save16bit = true;
% use zarr file as input, if false, use tiff as input
zarrFile = true;
% number of cpu cores
cpusPerTask = 14;
% use cluster computing for different images
parseCluster = false;
% set it to true for large files that cannot be fitted to RAM/GPU, it will
% split the data to chunks for deconvolution
largeFile = true;
% large method: "inplace": in place decon (only load the region with border buffer for decon);
%               "inmemory": for data can load to memory (load whole the
%               data, and decon a small region with border buffer). 
largeMethod = 'inplace';
% batch size to define each basic region for deconvolution (without
% including border buffer), typically as multipler of blockSize, and can
% fit to GPU if use GPU. 
% This size fits to GPUs with 24 GB or more vRAM for the demo data.
batchSize = [512, 768, 768];
% % block size to define the zarr chunk size (comment out to be consistent
% blockSize = [256, 256, 256];

% use GPU for deconvolution
GPUJob = true;
% if true, save intermediate results every 5 iterations.
debug = false;
% config file for the master jobs that runs on CPU node
ConfigFile = '';
% config file for the GPU job scheduling on GPU node
GPUConfigFile = '';
% if true, use Matlab runtime (for the situation without matlab license)
mccMode = false;

%% RL decon
XR_decon_data_wrapper(dataPaths_exp, 'deconPathstr', deconPathstr, 'xyPixelSize', xyPixelSize, ...   %'deconPathstr', deconPathstr not there. resultDirName defaults to matlab_decon
    'dz', dz, 'Reverse', Reverse, 'ChannelPatterns', ChannelPatterns, 'PSFFullpaths', PSFFullpaths, ...
    'dzPSF', dzPSF, 'parseSettingFile', parseSettingFile, 'RLmethod', RLmethod, ...
    'Background', Background , 'saveZarr', saveZarr,...
     'CPPdecon', false, 'CudaDecon', false, 'DeconIter', DeconIter, ...
    'fixIter', fixIter, 'EdgeErosion', EdgeErosion, 'Save16bit', Save16bit, ...
    'zarrFile', zarrFile, 'batchSize', batchSize, 'blockSize', blockSize, ...
    'parseCluster', parseCluster, 'largeFile', largeFile, 'largeMethod', largeMethod, ...
    'GPUJob', GPUJob, 'debug', debug, 'cpusPerTask', cpusPerTask, 'ConfigFile', ConfigFile, ...
    'GPUConfigFile', GPUConfigFile, 'mccMode', mccMode);
% release GPU if using GPU computing
if GPUJob && gpuDeviceCount('available') > 0
    reset(gpuDevice);
end

%%  large-scale deskew/rotation
dataPath_decon = [dataPaths_exp,deconPathstr,'\'];

% if true, use large scale processing pipeline (split, process, and then merge)
largeFile = true;
% true if input is in zarr format
zarrFile = true;
% save output as zarr if true
saveZarr = true;
% batch size for individual task, only the size in y is used, and it should
% be the multiplier of blocksize in y. Also need to adjust accordingly
% based on the available memory 
BatchSize = [512, 512, 512];
% block size to save the result 
blockSize = [256, 256, 256];
% save output as uint16 if true
Save16bit = true;

% use slurm cluster if true, otherwise use the local machine (master job)
parseCluster = false;
% use master job for task computing or not. 
masterCompute = true;
% configuration file for job submission
configFile = '';
% if true, use Matlab runtime (for the situation without matlab license)
mccMode = false;

XR_deskew_rotate_data_wrapper(dataPath_decon, xyPixelSize=xyPixelSize, dz=dz, ...
    ChannelPatterns=ChannelPatterns,Reverse=Reverse, largeFile=largeFile, zarrFile=zarrFile, saveZarr=saveZarr, ...
    BatchSize=BatchSize, blockSize=blockSize, Save16bit=Save16bit, parseCluster=parseCluster, ...
    masterCompute=masterCompute, configFile=configFile, mccMode=mccMode);

%% Test XR resample function
resample_factor  = [1 1 1]; % Update based on required downsample factor
for P_temp =dataPaths
    destPaths = [P_temp{1},'\matlab_stitch\matlab_decon\DSR'];
    % outPaths =  'Data_resample';
    % XR_resampleSingleZarr(destPaths,outPaths,resample_factor)
    zarr_im_lib = dir([destPaths '\*Abs.zarr']); 
        for j = 1:length(zarr_im_lib)
            fnin = [zarr_im_lib(j).folder '\' zarr_im_lib(j).name];
            fnout = [zarr_im_lib(j).folder '\' zarr_im_lib(j).name(1:end-5) '_resample_' num2str(resample_factor(1)) '_' num2str(resample_factor(2)) '_' num2str(resample_factor(3)) '.zarr'];
            XR_resampleSingleZarr(fnin,fnout,resample_factor)
            im_rep = readzarr(fnout);
            niftiwrite(im_rep, [fnout(1:end-5) '.nii']);
        end
end
