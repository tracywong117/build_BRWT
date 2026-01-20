#ifndef __VECTOR_SET_HPP__
#define __VECTOR_SET_HPP__

#include <vector>
#include <unordered_map>
#include <algorithm>

template <typename T,
          class Hash = std::hash<T>,
          class EqualTo = std::equal_to<T>,
          class Allocator = std::allocator<T>>
class VectorSet {
public:
    using value_type = T;
    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;

    std::pair<iterator, bool> insert(const T& val) {
        auto it = index_map_.find(val);
        if (it != index_map_.end()) {
            return {data_.begin() + it->second, false};
        }
        data_.push_back(val);
        index_map_.emplace(val, data_.size() - 1);
        return {data_.end() - 1, true};
    }
    
    // Simplistic emplace that just calls insert (copying/moving)
    template <typename... Args>
    std::pair<iterator, bool> emplace(Args&&... args) {
         return insert(T(std::forward<Args>(args)...));
    }

    iterator find(const T& val) {
        auto it = index_map_.find(val);
        if (it != index_map_.end()) {
            return data_.begin() + it->second;
        }
        return data_.end();
    }

    const_iterator find(const T& val) const {
        auto it = index_map_.find(val);
        if (it != index_map_.end()) {
            return data_.begin() + it->second;
        }
        return data_.end();
    }

    iterator begin() { return data_.begin(); }
    iterator end() { return data_.end(); }
    const_iterator begin() const { return data_.begin(); }
    const_iterator end() const { return data_.end(); }

    std::vector<T>& values_container() { return data_; }
    const std::vector<T>& values_container() const { return data_; }

    size_t size() const { return data_.size(); }

private:
    std::vector<T> data_;
    std::unordered_map<T, size_t, Hash, EqualTo> index_map_;
};

#endif // __VECTOR_SET_HPP__
