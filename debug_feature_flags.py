"""Debug feature flags to see what's happening."""

from testmaster.core.feature_flags import FeatureFlags

print("Initializing feature flags...")
FeatureFlags.initialize("testmaster_config.yaml")

print("Checking async processing flag...")
is_enabled = FeatureFlags.is_enabled('layer2_monitoring', 'async_processing')
print(f"async_processing enabled: {is_enabled}")

print("Checking layer2_monitoring file_monitoring...")
layer2_enabled = FeatureFlags.is_enabled('layer2_monitoring', 'file_monitoring')
print(f"layer2_monitoring file_monitoring enabled: {layer2_enabled}")

print("Getting all feature flags...")
all_flags = FeatureFlags.get_all_features()
print(f"All flags: {all_flags}")