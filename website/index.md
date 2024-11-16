---
datatable: true
---


<link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.21/css/jquery.dataTables.min.css" />
<script src="https://code.jquery.com/jquery-3.5.1.js"></script>
<script src="https://cdn.datatables.net/1.10.21/js/jquery.dataTables.min.js"></script>


TabularBench: Adversarial robustness benchmark for tabular data.

[Documentation](https://serval-uni-lu.github.io/tabularbench/doc).

## Leaderboard

You are currently viewing results of the leaderboard. View MOEVA attack results [here](moeva).

Jump to dataset:

- [CTU](#ctu)
- [LCLD](#lcld)
- [Malware](#malware)
- [URL](#url)
- [WIDS](#wids)

### CTU

<a href="#">^ back to top</a>

{% include_relative tables/ctu.md %}

### LCLD

<a href="#">^ back to top</a>

{% include_relative tables/lcld.md %}

### Malware

<a href="#">^ back to top</a>

{% include_relative tables/malware.md %}

### URL

<a href="#">^ back to top</a>

{% include_relative tables/url.md %}

### WIDS

<a href="#">^ back to top</a>

{% include_relative tables/wids.md %}

<script>
    var table = $('table').DataTable(
        {
            "bPaginate": false,
            "language": {
                searchPlaceholder: 'Architectures, training methods, etc.'
            },
            // "autoWidth": true,
        }
    );
    table.columns.adjust().draw();

    $(document).ready(function () {
        function updateFilterMargins() {
            $('.dataTables_wrapper').each(function () {
                var $wrapper = $(this);
                var $dataTable = $wrapper.find('.dataTable'); // Find the dataTable within the wrapper
                var $filter = $wrapper.find('.dataTables_filter'); // Find the filter within the wrapper
                if ($dataTable.length && $filter.length) {
                    // Get the computed right margin of the dataTable
                    var tableMarginRight = parseFloat($dataTable.css('margin-right')) || 0;
                    // Apply the same margin to the filter
                    $filter.css('margin-right', tableMarginRight);
                }
            });
        }

        // Update margins initially
        updateFilterMargins();

        // Listen for resize events on the window to update margins dynamically
        $(window).on('resize', function () {
            updateFilterMargins();
        });
    });
</script>
