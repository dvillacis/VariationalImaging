__precompile__()

module TestDatasets

using FileIO
using Pkg.Artifacts
using StringDistances

const artifacts_toml = abspath(joinpath(@__DIR__, "..", "Artifacts.toml"))

export testdataset

const remotedatasets = [
    "cameraman",
    "mandrill",
    "peppers"
]

function testdataset(datasetname; download_only::Bool = false)

    datasetfiles = dataset_path(full_datasetname(datasetname))
    println(datasetfiles)
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
    ensure_artifact_installed("images", artifacts_toml)

    dataset_dir = artifact_path(artifact_hash("images", artifacts_toml))
    return joinpath(dataset_dir, datasetname)
end

_findall(name; min_score=0.6) = findall(name, remotefiles,JaroWinkler(), min_score=min_score)
_findnearest(name; min_score=0.8) = findnearest(name, remotefiles, JaroWinkler(), min_score=min_score)

end