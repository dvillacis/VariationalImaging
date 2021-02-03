__precompile__()

module TestDatasets

using FileIO
using Pkg.Artifacts
using StringDistances

const artifacts_toml = abspath(joinpath(@__DIR__, "..", "Artifacts.toml"))

export testdataset

const remotedatasets = [
    "cameraman",
    "cameraman_128",
    "mandrill",
    "peppers"
]

function testdataset(datasetname; download_only::Bool = false)

    datasetfiles = dataset_path(full_datasetname(datasetname))
    download_only && return datasetfiles
    dataset = load_dataset(datasetfiles)
    return dataset

end

function full_datasetname(datasetname)
    idx = findfirst(remotedatasets) do x
        startswith(x, datasetname)
    end
    if idx === nothing
        warn_msg = "\"$datasetname\" not found in `TestDatasets.remotefiles`."

        best_match = _findnearest(datasetname)
        if isnothing(best_match[2])
            similar_matches = remotefiles[_findall(datasetname)]
            if !isempty(similar_matches)
                similar_matches_msg = "  * \"" * join(similar_matches, "\"\n  * \"") * "\""
                warn_msg = "$(warn_msg) Do you mean one of the following?\n$(similar_matches_msg)"
            end
            throw(ArgumentError(warn_msg))
        else
            idx = best_match[2]
            @warn "$(warn_msg) Load \"$(remotefiles[idx])\" instead."
        end
    end
    return remotedatasets[idx]
end

function dataset_path(datasetname)
    ensure_artifact_installed("datasets", artifacts_toml)

    dataset_dir = artifact_path(artifact_hash("datasets", artifacts_toml))
    return joinpath(dataset_dir, "datasets" ,datasetname)
end

function load_dataset(datasetfiles)
    image_pairs = readlines(joinpath(datasetfiles,"filelist.txt"))
    M,N = size(load(joinpath(datasetfiles,split(image_pairs[1],",")[1])))
    true_images = zeros(M,N,length(image_pairs))
    data_images = zeros(M,N,length(image_pairs))
    for i = 1:length(image_pairs)
        pair = split(image_pairs[i],",")
        true_images[:,:,i] = load(joinpath(datasetfiles,pair[1]))
        data_images[:,:,i] = load(joinpath(datasetfiles,pair[2]))
    end
    return true_images,data_images
end

_findall(name; min_score=0.6) = findall(name, remotefiles,JaroWinkler(), min_score=min_score)
_findnearest(name; min_score=0.8) = findnearest(name, remotefiles, JaroWinkler(), min_score=min_score)

end