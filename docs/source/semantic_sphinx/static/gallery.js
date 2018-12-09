var Gallery = {
    examples: null,
    contents: null,
    categories: null,

    drawExample: function (key) {
        var example = this.examples[key]

        var image_div = $('<div>', {
            class: 'image'
        }).append($('<img>', {
            src: "../_static/" + example.thumb
        }))

        var contents_div = $('<div>', {
            class: 'content'
        }).append($('<div>', {
            class: 'header'
        }).text(example.title))

        var div = $('<a>', {
            class: 'card',
            href: example.url
        }).append(image_div).append(contents_div)
        return div
    },

    makeExamples: function (examples) {
        var cards = $("<div>", {
            class: "ui link six stackable cards"
        })
        for (var j = 0; j < examples.length; j++) {
            cards.append(this.drawExample(examples[j]))
        }
        return cards
    },

    drawExamples: function () {
        var main_div = $("#gallery")
        var gallery = this;
        var categories = this.getCategories()
        var cats = Object.keys(categories)
        cats.sort()

        cats.map(function (category) {
            var div = $("<div>", {
                class: "ui vertical segment"
            })
            div.append($("<h2>", {
                class: "ui header"
            }).text(category))
            div.append(gallery.makeExamples(categories[category]))
            main_div.append(div)
        })
    },

    getCategories: function () {
        var categories = {};
        var gallery = this;
        var uniqueCategories = Array.from(new Set(Object.values(this.contents)))
        for (var i in uniqueCategories) {
            categories[uniqueCategories[i]] = []
        }
        categories["Other"] = []
        Object.keys(this.examples).forEach(function (key) {
            if (key in gallery.contents) {
                categories[gallery.contents[key]].push(key)
            }
            else {
                categories["Other"].push(key)
            }
        })
        return categories
    },

    loadScript: function (url, eltId) {
        var self = this;
        $.ajax({
            type: "GET", url: url, data: null,
            dataType: "script", cache: true,
            complete: function (jqxhr, textstatus) {
                if (textstatus != "success") {
                    document.getElementById(eltId).src = url;
                }
                self.drawExamples();
            }
        });
    }

}
